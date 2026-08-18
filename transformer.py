import math
class Tensor:
    def __init__(self,data):
        self.data=data
        self.shape=self._get_shape(data)
        
    def _get_shape(self,data):
        if isinstance(data, list):
            if len(data)==0:
                return(0,)
            return (len(data),)+self._get_shape(data[0])
        return ()
        
    def __repr__(self):
        return f"Tensor(shape={self.shape},data={self.data})"
        
    def __matmul__(self,other):
        if len(self.shape)!=2 or len(other.shape)!=2:
            raise ValueError("Matrix multiplication requires two 2D tensors.")
            
        rows_a, cols_a=self.shape
        rows_b, cols_b=other.shape
        
        if cols_a!=rows_b:
            raise ValueError(f"Cannot multiply matrices with shape {self.shape} and {other.shape}")
            
        result =[[0 for _ in range(cols_b)] for _ in range(rows_a)]
        
        for i in range(rows_a):
            for j in range(cols_b):
                for k in range(cols_a):
                    result[i][j]+=self.data[i][k]*other.data[k][j]
                    
        return Tensor(result)
        
    def __add__(self, other):
    # Case 1: Same shape
        if self.shape == other.shape:
            rows, cols = self.shape

            result = [[0 for _ in range(cols)] for _ in range(rows)]

            for i in range(rows):
                for j in range(cols):
                    result[i][j] = self.data[i][j] + other.data[i][j]

            return Tensor(result)

        # Case 2: Broadcasting (1, n) across (m, n)
        if (
            len(self.shape) == 2
            and len(other.shape) == 2
            and other.shape[0] == 1
            and self.shape[1] == other.shape[1]
        ):
            rows, cols = self.shape

            result = [[0 for _ in range(cols)] for _ in range(rows)]

            for i in range(rows):
                for j in range(cols):
                    result[i][j] = self.data[i][j] + other.data[0][j]

            return Tensor(result)

        raise ValueError(
            f"Cannot add tensors with shapes {self.shape} and {other.shape}"
        )
        
    def transpose(self):
        rows,cols=self.shape
        result=[[0 for _ in range(rows)] for _ in range(cols)]
        
        for i in range(rows):
            for j in range(cols):
                result[j][i]=self.data[i][j]
                
        return Tensor(result)
        
    def __mul__(self, scalar):
        rows, cols = self.shape

        result = [[0 for _ in range(cols)] for _ in range(rows)]

        for i in range(rows):
            for j in range(cols):
                result[i][j] = self.data[i][j] * scalar

        return Tensor(result)

    def __truediv__(self, scalar):
        rows, cols = self.shape

        result = [[0 for _ in range(cols)] for _ in range(rows)]

        for i in range(rows):
            for j in range(cols):
                result[i][j] = self.data[i][j] / scalar

        return Tensor(result)
        
    def cat(self, other):
        if self.shape[0] != other.shape[0]:
            raise ValueError(
                f"Cannot concatenate tensors with row counts "
                f"{self.shape[0]} and {other.shape[0]}"
            )

        rows = self.shape[0]

        result = []

        for i in range(rows):
            result.append(self.data[i] + other.data[i])

        return Tensor(result)
    
        
class Module:
    def forward(self, x):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)
        
class Linear(Module):
    def __init__(self, in_features, out_features):

        # Small constant weights
        self.weight = Tensor([
            [0.01 for _ in range(out_features)]
            for _ in range(in_features)
        ])

        # Bias row
        self.bias = Tensor([
            [0.0 for _ in range(out_features)]
        ])

    def forward(self, x):
        return (x @ self.weight) + self.bias
    
    
class Embedding(Module):
    def __init__(self, vocab_size, embedding_dim):

        self.weight = Tensor([
            [(i + 1) * 0.01 for _ in range(embedding_dim)]
            for i in range(vocab_size)
        ])

    def forward(self, x):
        result = []

        for token_id in x:
            result.append(self.weight.data[token_id])

        return Tensor(result)
        
import math

class Softmax(Module):
    def __init__(self):
        pass

    def forward(self, x):
        result = []

        # Process each row independently
        for row in x.data:

            # Step 1: Find maximum value
            row_max = max(row)

            # Step 2: Compute exponentials
            exps = []
            for value in row:
                exps.append(math.exp(value - row_max))

            # Step 3: Sum exponentials
            exp_sum = sum(exps)

            # Step 4: Normalize
            softmax_row = []
            for exp_val in exps:
                softmax_row.append(exp_val / exp_sum)

            result.append(softmax_row)

        return Tensor(result)
        
class SelfAttention(Module):
    def __init__(self, embed_dim, head_dim):

        self.embed_dim = embed_dim
        self.head_dim = head_dim

        self.q_proj = Linear(embed_dim, head_dim)
        self.k_proj = Linear(embed_dim, head_dim)
        self.v_proj = Linear(embed_dim, head_dim)
        
        self.softmax = Softmax()
        

    def forward(self, x):

        # Step 1: Generate Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Step 2: Attention scores
        scores = Q @ K.transpose()

        # Step 3: Scale scores
        scores = scores / math.sqrt(self.head_dim)

        # Step 3.5: Causal Masking
        rows, cols = scores.shape

        for i in range(rows):
            for j in range(cols):
                if j > i:
                    scores.data[i][j] = -float('inf')

        # Step 4: Softmax
        weights = self.softmax(scores)

        # Step 5: Weighted sum
        output = weights @ V

        return output
        
        
class MultiHeadAttention(Module):
    def __init__(self, num_heads, embed_dim, head_dim):

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = head_dim

        self.heads = [
            SelfAttention(embed_dim, head_dim)
            for _ in range(num_heads)
        ]

        self.o_proj = Linear(
            num_heads * head_dim,
            embed_dim
        )

    def forward(self, x):

        head_outputs = []

        for head in self.heads:
            head_outputs.append(head(x))

        combined = head_outputs[0]

        for i in range(1, len(head_outputs)):
            combined = combined.cat(head_outputs[i])

        output = self.o_proj(combined)

        return output
        
class ReLU(Module):

    def forward(self, x):

        rows, cols = x.shape

        result = [[0 for _ in range(cols)] for _ in range(rows)]

        for i in range(rows):
            for j in range(cols):
                result[i][j] = max(0.0, x.data[i][j])

        return Tensor(result)
        
class FeedForward(Module):

    def __init__(self, embed_dim):

        self.fc1 = Linear(
            embed_dim,
            4 * embed_dim
        )

        self.relu = ReLU()

        self.fc2 = Linear(
            4 * embed_dim,
            embed_dim
        )

    def forward(self, x):

        x = self.fc1(x)

        x = self.relu(x)

        x = self.fc2(x)

        return x
    
import math

class LayerNorm(Module):

    def __init__(self, embed_dim):
        self.embed_dim = embed_dim

    def forward(self, x):

        result = []

        for row in x.data:

            # Mean
            mean = sum(row) / len(row)

            # Variance
            variance = 0.0

            for value in row:
                variance += (value - mean) ** 2

            variance /= len(row)

            # Normalize
            normalized_row = []

            for value in row:
                normalized_value = (
                    (value - mean)
                    / math.sqrt(variance + 1e-5)
                )

                normalized_row.append(normalized_value)

            result.append(normalized_row)

        return Tensor(result)
        
class TransformerBlock(Module):

    def __init__(self, embed_dim, num_heads, head_dim):

        self.ln1 = LayerNorm(embed_dim)

        self.attn = MultiHeadAttention(
            num_heads,
            embed_dim,
            head_dim
        )

        self.ln2 = LayerNorm(embed_dim)

        self.ffn = FeedForward(embed_dim)

    def forward(self, x):

        x = x + self.attn(self.ln1(x))

        x = x + self.ffn(self.ln2(x))

        return x
    
    
class Transformer(Module):

    def __init__(
        self,
        vocab_size,
        max_seq_len,
        embed_dim,
        num_heads,
        head_dim,
        num_layers
    ):

        # Token embeddings
        self.token_emb = Embedding(
            vocab_size,
            embed_dim
        )

        # Positional embeddings
        self.pos_emb = Embedding(
            max_seq_len,
            embed_dim
        )

        # Transformer blocks
        self.blocks = [
            TransformerBlock(
                embed_dim,
                num_heads,
                head_dim
            )
            for _ in range(num_layers)
        ]

        # Final normalization
        self.ln_f = LayerNorm(embed_dim)

        # Language modeling head
        self.lm_head = Linear(
            embed_dim,
            vocab_size
        )

    def forward(self, x):

        # Token embeddings
        token_embeddings = self.token_emb(x)

        # Position IDs
        positions = list(range(len(x)))

        # Position embeddings
        position_embeddings = self.pos_emb(positions)

        # Combine token + position embeddings
        h = token_embeddings + position_embeddings

        # Pass through all transformer blocks
        for block in self.blocks:
            h = block(h)

        # Final layer norm
        h = self.ln_f(h)

        # Project to vocabulary logits
        logits = self.lm_head(h)

        return logits
    
def generate(model, input_tokens, max_new_tokens):

    for _ in range(max_new_tokens):

        # Get model predictions
        logits = model(input_tokens)

        # Last token's logits
        last_row = logits.data[-1]

        # Find token with highest score
        next_token = last_row.index(max(last_row))

        # Append generated token
        input_tokens.append(next_token)

    return input_tokens
    
if __name__ == '__main__':

    # Hyperparameters
    vocab_size = 100
    max_seq_len = 32
    embed_dim = 16

    num_heads = 4
    head_dim = 4

    num_layers = 2

    # Create model
    model = Transformer(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        num_layers=num_layers
    )

    # Starting sequence
    input_seq = [5, 10, 15]

    # Generate 5 new tokens
    output = generate(
        model,
        input_seq,
        max_new_tokens=5
    )

    print("Generated sequence:")
    print(output)
    
    
    
    
    
    
    
    
    
    
    
    
    
    