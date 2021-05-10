setClass('SelfAttention', slots=list(emb='numeric', heads='numeric', mask='logical'))






self <- list()

SelfAttention <- function(emb, heads=8, mask=FALSE) {
  #
  # Canonical implementation of multi-head self attention.
  #

  stopifnot(emb %% heads, 0)

  s = emb / heads

  # self$tokeys =
  # self$toqueries =
  # self$tovalues =
  #
  # self$unifyheads =

}


foward <- function(self, x) {
  # b, t, e = x.size()
  h = self$heads

  stopifnot(e, self$emb)

  s = e / h

  # keys    = self$tokeys
  # queries = self$toqueries
  # values  = self$tovalues

  # keys    = keys.view(b, t, h, s)
  # queries = queries.view(b, t, h, s)
  # values  = values.view(b, t, h, s)


  # -- We first compute the k/q/v's on the whole embedding vectors, and then split into the different heads.
  #    See the following video for an explanation: https://youtu.be/KmAISyVvE1Y

  # Compute scaled dot-product self-attention

  # # - fold heads into the batch dimension
  # keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
  # queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
  # values = values.transpose(1, 2).contiguous().view(b * h, t, s)
  #
  # queries = queries / (e ** (1/4))
  # keys    = keys / (e ** (1/4))
  # # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
  # #   This should be more memory efficient
  #
  # # - get dot product of queries and keys, and scale
  # dot = torch.bmm(queries, keys.transpose(1, 2))
  #
  # assert dot.size() == (b*h, t, t)
  #
  # if self.mask: # mask out the upper half of the dot matrix, excluding the diagonal
  #   mask_(dot, maskval=float('-inf'), mask_diagonal=False)
  #
  # dot = F.softmax(dot, dim=2)
  # # - dot now has row-wise self-attention probabilities
  #
  # # apply the self attention to the values
  # out = torch.bmm(dot, values).view(b, h, t, s)
  #
  # # swap h, t back, unify heads
  # out = out.transpose(1, 2).contiguous().view(b, t, s * h)
  #
  # return self.unifyheads(out)
}

# class TransformerBlock(nn.Module):
#
#   def __init__(self, emb, heads, mask, seq_length, ff_hidden_mult=4, dropout=0.0, attention_type='default'):
#   super().__init__()
#
# if attention_type == 'default':
#   self.attention = SelfAttention(emb, heads=heads, mask=mask)
# elif attention_type == 'wide':
#   self.attention = SelfAttentionWide(emb, heads=heads, mask=mask)
# elif attention_type == 'gpt2':
#   self.attention = SelfAttentionGPT2(emb, heads=heads, mask=mask)
# elif attention_type == 'narrow':
#   self.attention = SelfAttentionNarrow(emb, heads=heads, mask=mask)
# else:
#   raise Exception(f'Self-attention type {type} not recognized.')
#
# self.mask = mask
#
# self.norm1 = nn.LayerNorm(emb)
# self.norm2 = nn.LayerNorm(emb)
#
# self.ff = nn.Sequential(
#   nn.Linear(emb, ff_hidden_mult * emb),
#   nn.ReLU(),
#   nn.Linear(ff_hidden_mult * emb, emb)
# )
#
# self.do = nn.Dropout(dropout)
#
# def forward(self, x):
#
#   attended = self.attention(x)
#
# x = self.norm1(attended + x)
#
# x = self.do(x)
#
# fedforward = self.ff(x)
#
# x = self.norm2(fedforward + x)
#
# x = self.do(x)
#
# return x

