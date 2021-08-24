import numpy as np
import torch

import tensor_buffer as tb

class Reducer:
    def __init__(self, random_seed, device):
        self.rng = np.random.RandomState(random_seed)
        M = 1024 * 1024
        self.precalc_numbers = (
            torch.from_numpy(self.rng.randn(128 * M)).to(device).type(torch.float32)
        )
        if torch.distributed.is_available():
            self.n_workers = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.n_workers = 1
            self.rank = 0
        self.device = device

    def reduce(self, grad_in, grad_out, memory_out):
        """Return communicated bits"""
        raise NotImplementedError()

class PowerSGDReducer(Reducer):
    def __init__(self, random_seed, device, n_power_iterations=0, reuse_query=True, rank=1):
        # reuse is important for PowerSGD ... so reuse_query=True
        super().__init__(random_seed, device)
        # check if power iteration == 0 or not
        assert n_power_iterations == 0
        self.rank = rank
        self.p_memory = None
        self.q_memory = None
        self.reuse_query = reuse_query

    def _set_random(self, vector):
        torch.manual_seed(self.rng.randint(1_000_000_000))
        vector.data[:] = torch.randn(*vector.shape, device=self.device)
        # orthogonalize needs to be done
        # But almost not needed... randn make almost perfect
        # orthogonalize(vector)

    def reduce(self, grad_in, grad_out, memory_out):
        """
        grad_in, grad_out, memory_out : dictionary of params grads
        return total communicated
        """
        bits_communicated = 0

        # [For Rank 1] It's out of Algorithms!!!!
        # rank1 tensors will be reduced un-compressed
        # and rank > 1 tensors should be compressed and reduced
        rank1_tensors = [
            (tensor, out, mem)
            for tensor, out, mem in zip(grad_in, grad_out, memory_out)
            if tensor.ndimension() <= 1
        ]
        high_rank_tensors = [
            (tensor, out, mem)
            for tensor, out, mem in zip(grad_in, grad_out, memory_out)
            if tensor.ndimension() > 1
        ]

        # build rank-k approx of every tensor
        # Approx equation
        # M = PQ^T
        # allocate consequtive mem for P's and Q's

        mem_uninitialized = self.p_memory is None

        # Step 1. Calc Matrix Size and Allocate Memory
        p_total_size = 0
        q_total_size = 0
        for tensor, _, _ in high_rank_tensors:
            # convert grad(M) into 2d tensor
            matrix = tensor.view(tensor.shape[0], -1)
            n, m = matrix.shape
            rank = min(n, m, self.rank)
            p_total_size += n*rank
            q_total_size += m*rank
        # [Important] Initialization on Device !!!
        if self.p_memory == None: # not initialized
            self.p_memory = torch.empty(p_total_size, device=self.device)
            self.q_memory = torch.empty(q_total_size, device=self.device)
        # for easier implementation, gather pointers
        p_ptrs = []
        q_ptrs = []
        p_idx = 0
        q_idx = 0
        for tensor, _, _ in high_rank_tensors:
            matrix = tensor.view(tensor.shape[0], -1)
            n, m = matrix.shape
            rank = min(n, m , self.rank)
            # torch.tensor.view returns pointer
            p_ptrs.append(self.p_memory[p_idx : p_idx + n*rank].view(n, rank))
            q_ptrs.append(self.q_memory[q_idx : q_idx + m*rank].view(m, rank))
            p_idx += n * rank
            q_idx += m * rank

        # Step 2. Prepare Q if not initailized
        for (tensor, _, _), q, p in zip(high_rank_tensors, q_ptrs, p_ptrs):
            matrix = tensor.view(tensor.shape[0], -1)
            n, m = matrix.shape
            if self.reuse_query and not mem_uninitialized:
                # if u wanna reuse and already init
                # use prev_Q
                # do not need orthogonalize if properly _set_random...ed!
                # orthogonalize(q)
                pass
            else:
                self._set_random(q)
        
        """
        PowerSGD
        Algorithm 1: Rank-r PowerSGD Compression
        
        All Compression/Decompression is done in Reducer
        """

        # Step 3. (Algo 1: line 3) P <- MQ (Compute P)
        for (tensor, _, _), q, p in zip(high_rank_tensors, q_ptrs, p_ptrs):
            matrix = tensor.view(tensor.shape[0], -1)
            torch.matmul(matrix, q, out=p)
        
        # Step 4. (Algo 1: line 4) ALL_REDUCE_MEAN(P)
        all_reduce(self.p_memory)
        bits_communicated += n_bits(self.p_memory)
        self.p_memory.data[:] /= self.n_workers

        # [For Rank 1] Start Communicating Rank 1 Tensors
        rank1_tensor_list = tb.TensorBuffer([tensor for (tensor, _, _) in rank1_tensors])
        rank1_handler = rank1_tensor_list.all_reduce(async_op=True)
        bits_communicated += rank1_tensor_list.bits()

        # Step 5. (Algo 1: line 5) P_hat <- ORTHOGONALIZE(P)
        for p in p_ptrs:
            orthogonalize(p)

        # Step 6. (Algo 1: line 6) Q <- M_T P_hat
        for p, q, (tensor, _, _) in zip(p_ptrs, q_ptrs, high_rank_tensors):
            matrix = tensor.view(tensor.shape[0], -1)
            torch.matmul(matrix.t(), p, out=q)
        
        # Step 7. (Algo 1: line 7) ALL_REDUCE_MEAN(Q)
        all_reduce(self.q_memory)
        bits_communicated += n_bits(self.q_memory)
        self.q_memory.data[:] /= self.n_workers

        """
        PowerSGD
        Algorithm 2: Distributed Error-feedback SGD with Momentum

        Only Local Error is return by Reducer!
        Main Algorithm is implemented in Main Process
        """

        # Step 8. (Algo 1: line 11) Decompress
        for p, q, (tensor, out, mem) in zip(p_ptrs, q_ptrs, high_rank_tensors):
            # einsum representation
            # out.data[:] = torch.einsum("nr, mr -> nm", (p, q)).view(*tensor.shape)
            torch.matmul(p, q.t(), out=out.data[:])
            # Step 9. (Algo 2: line 9) Memorize Local Errors
            mem.data[:] = tensor - out

        # [For Rank 1] Wait for Reducing
        rank1_handler.wait()
        rank1_tensor_list.buffer /= self.n_workers
        rank1_tensor_list.unpack([out for (_, out, _) in rank1_tensors])

        return bits_communicated





        


    
@torch.jit.script
def orthogonalize(matrix, eps=torch.tensor(1e-8)):
    n, m = matrix.shape
    for i in range(m):
        # Normalize the i'th column
        col = matrix[:, i : i + 1]
        col /= torch.sqrt(torch.sum(col ** 2)) + eps
        # Project it on the rest and remove it
        if i + 1 < m:
            rest = matrix[:, i + 1 :]
            # rest -= torch.matmul(col.t(), rest) * col
            rest -= torch.sum(col * rest, dim=0) * col

def all_reduce(*args, **kwargs):
    if torch.distributed.is_available() and torch.distributed.get_world_size() > 1:
        return torch.distributed.all_reduce(*args, **kwargs)

def n_bits(tensor):
    return 8 * tensor.nelement() * tensor.element_size()