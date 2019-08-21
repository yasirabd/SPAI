# Machine Translation with a Sequence to Sequence Network and Attention (English - Indonesian)
Create machine translation from English to Indonesian

# Step-by-step
## Step 1: Install requirements
- Installing Pytorch and Syft.

## Step 2: Loading Data Files
- Download Tatoeba dataset <code>eng->Indonesian</code>.
- Do preprocessing:
    - Read text file and split into lines, split lines into pairs
    - Normalize text, filter by length and content
    - Make word lists from sentences in pairs
- Transform dataset into <code>input_lang</code> for encoder, <code>output_lang</code> for decoder, and <code>pairs</code> for training model.


## Step 3: The Seq2Seq Model
- Create EncoderRNN: The encoder of a seq2seq network is a RNN that outputs some value for every word from the input sentence. For every input word the encoder outputs a vector and a hidden state, and uses the hidden state for the next input word.

![EncoderRNN](https://github.com/yasirabd/SPAI/blob/master/Project%203/assets/encoder-network.png "EncoderRNN")

- Create DecoderRNN: The decoder is another RNN that takes the encoder output vector(s) and outputs a sequence of words to create the translation

![DecoderRNN](https://github.com/yasirabd/SPAI/blob/master/Project%203/assets/decoder-network.png "DecoderRNN")

- Create Attention Decoder: Attention allows the decoder network to “focus” on a different part of the encoder’s outputs for every step of the decoder’s own outputs.

![DecoderAttn](https://github.com/yasirabd/SPAI/blob/master/Project%203/assets/attention-decoder-network.png "DecoderAttn")

## Step 4: Training
- To train we run the input sentence through the encoder, and keep track of every output and the latest hidden state. Then the decoder is given the token as its first input, and the last hidden state of the encoder as its first hidden state.

## Step 5: Evaluation
- Evaluation is mostly the same as training, but there are no targets so we simply feed the decoder’s predictions back to itself for each step.
- The result such as follows:
```
> saya malu telah melakukan hal tersebut .
= i am ashamed of having done so .
< i am ashamed of having done so . <EOS>

> saya senang bisa membantu .
= i m glad i could help out .
< i m glad i could help . <EOS>

> dia pintar .
= he s smart .
< he s intelligent . <EOS>

> aku minta maaf sudah mengecewakanmu .
= i m sorry i let you down .
< i m sorry i let you down <EOS>

> aku lebih pendek ketimbang kamu .
= i am shorter than you .
< i m shorter than you . <EOS>

> saya cukup nyaman di ruangan ini .
= i am quite comfortable in this room .
< i am quite comfortable in this room . <EOS>

> aku orang yang rasional .
= i m reasonable .
< i m reasonable . <EOS>

> saya senang bisa membantu .
= i m glad i could help out .
< i m glad i could help . <EOS>

> dia takut kucing .
= she is afraid of cats .
< she is afraid of cats . <EOS>

> aku tidak sedang bercanda .
= i am not kidding .
< i m not kidding . <EOS>
```
- Evaluate and showing Attention
```python
evaluateAndShowAttention("saya sangat lapar .")
> input = saya sangat lapar .
> output = i am terribly hungry . <EOS>
```
