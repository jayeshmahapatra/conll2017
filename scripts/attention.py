from __future__ import print_function
from sys import argv, stderr
import random

import dynet as dy

from augment import augment

EOS = "<EOS>"

LSTM_NUM_OF_LAYERS = 2
EMBEDDINGS_SIZE = 32
STATE_SIZE = 32
ATTENTION_SIZE = 32


def embed_sentence(sentence):
    sentence = [EOS] + list(sentence) + [EOS]
    # Skip unknown characters.
    sentence = [char2int[c] for c in sentence if c in char2int]

    global input_lookup

    return [input_lookup[char] for char in sentence]


def run_lstm(init_state, input_vecs):
    s = init_state

    out_vectors = []
    for vector in input_vecs:
        s = s.add_input(vector)
        out_vector = s.output()
        out_vectors.append(out_vector)
    return out_vectors


def encode_sentence(enc_fwd_lstm, enc_bwd_lstm, sentence):
    sentence_rev = list(reversed(sentence))

    fwd_vectors = run_lstm(enc_fwd_lstm.initial_state(), sentence)
    bwd_vectors = run_lstm(enc_bwd_lstm.initial_state(), sentence_rev)
    bwd_vectors = list(reversed(bwd_vectors))
    vectors = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]

    return vectors


def attend(input_mat, state, w1dt):
    global attention_w2
    global attention_v
    w2 = dy.parameter(attention_w2)
    v = dy.parameter(attention_v)

    # input_mat: (encoder_state x seqlen) => input vecs concatenated as cols
    # w1dt: (attdim x seqlen)
    # w2dt: (attdim x attdim)
    w2dt = w2*dy.concatenate(list(state.s()))
    # att_weights: (seqlen,) row vector
    unnormalized = dy.transpose(v * dy.tanh(dy.colwise_add(w1dt, w2dt)))
    att_weights = dy.softmax(unnormalized)
    # context: (encoder_state)
    context = input_mat * att_weights
    return context


def decode(dec_lstm, vectors, output):
    output = [EOS] + list(output) + [EOS]
    output = [char2int[c] for c in output]

    w = dy.parameter(decoder_w)
    b = dy.parameter(decoder_b)
    w1 = dy.parameter(attention_w1)
    input_mat = dy.concatenate_cols(vectors)
    w1dt = None

    last_output_embeddings = output_lookup[char2int[EOS]]
    s = dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(STATE_SIZE*2), last_output_embeddings]))
    loss = []

    for char in output:
        # w1dt can be computed and cached once for the entire decoding phase
        w1dt = w1dt or w1 * input_mat
        vector = dy.concatenate([attend(input_mat, s, w1dt), last_output_embeddings])
        s = s.add_input(vector)
        out_vector = w * s.output() + b
        probs = dy.softmax(out_vector)
        last_output_embeddings = output_lookup[char]
        loss.append(-dy.log(dy.pick(probs, char)))
    loss = dy.esum(loss)
    return loss


def generate(in_seq, enc_fwd_lstm, enc_bwd_lstm, dec_lstm):
    embedded = embed_sentence(in_seq)
    encoded = encode_sentence(enc_fwd_lstm, enc_bwd_lstm, embedded)

    w = dy.parameter(decoder_w)
    b = dy.parameter(decoder_b)
    w1 = dy.parameter(attention_w1)
    input_mat = dy.concatenate_cols(encoded)
    w1dt = None

    last_output_embeddings = output_lookup[char2int[EOS]]
    s = dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(STATE_SIZE * 2), last_output_embeddings]))

    out = ''
    count_EOS = 0
    for i in range(len(in_seq)*2):
        if count_EOS == 2: break
        # w1dt can be computed and cached once for the entire decoding phase
        w1dt = w1dt or w1 * input_mat
        vector = dy.concatenate([attend(input_mat, s, w1dt), last_output_embeddings])
        s = s.add_input(vector)
        out_vector = w * s.output() + b
        probs = dy.softmax(out_vector).vec_value()
        next_char = probs.index(max(probs))
        last_output_embeddings = output_lookup[next_char]
        if int2char[next_char] == EOS:
            count_EOS += 1
            continue

        out += int2char[next_char]
    return out


def get_loss(input_sentence, output_sentence, enc_fwd_lstm, enc_bwd_lstm, dec_lstm):
    dy.renew_cg()
    embedded = embed_sentence(input_sentence)
    encoded = encode_sentence(enc_fwd_lstm, enc_bwd_lstm, embedded)
    return decode(dec_lstm, encoded, output_sentence)


def train(model, isentences,osentences, idevsentences,odevsentences,itestsentences, ofile):
    trainer = dy.SimpleSGDTrainer(model,e0=.01)
    iopairs = list(zip(isentences,osentences))
    random.shuffle(iopairs)
    for i in range(EPOCHS):
        loss_value = 0
#        random.shuffle(iopairs)
        for isentence, osentence in iopairs:
            loss = get_loss(isentence, osentence, enc_fwd_lstm, enc_bwd_lstm, dec_lstm)
            loss_value += loss.value()
            loss.backward()
            trainer.update()

        corr = 0
        for ip,op in zip(idevsentences,odevsentences):
            dy.renew_cg()
            sys_o = generate(ip, enc_fwd_lstm, enc_bwd_lstm, dec_lstm)
            if ''.join(op) == sys_o:
                corr += 1
        print("EPOCH %u: LOSS %.2f, DEV ACC %.2f" % (i+1, loss_value/len(iopairs), corr * 100.0 / len(idevsentences)))
        
    for ilemma,ilabels in itestsentences:
        dy.renew_cg()
        sys_o = generate(ilemma+ilabels, enc_fwd_lstm, enc_bwd_lstm, dec_lstm)
        ofile.write("%s\t%s\t%s\n" % (''.join(ilemma),sys_o,';'.join(ilabels)))

if __name__=='__main__':
    # This is so ugly.
    global characters, char2int, int2char, EPOCHS

    if len(argv) != 7:
        stderr.write(("USAGE: python3 %s train_file dev_file "+
                      "test_file aug_factor num_epochs ofile\n") % argv[0])
        exit(1)

    TRAIN_FN=argv[1]
    DEV_FN=argv[2]
    TEST_FN=argv[3]
    AUG_FACTOR=int(argv[4])
    EPOCHS=int(argv[5])
    O_FN = argv[6]

    data = augment([l.strip().split('\t') for l in open(TRAIN_FN).read().split('\n') if l.strip() != ''],AUG_FACTOR)
    
    idata = [[c for c in lemma] + tags.split(';') for lemma, _, tags in data]
    odata = [[c for c in wf] for _, wf, _ in data]
    
    devdata = [l.strip().split('\t') for l in open(DEV_FN).read().split('\n') if l.strip() != '']
    idevdata = [[c for c in lemma] + tags.split(';') for lemma, _, tags in devdata]
    odevdata = [[c for c in wf] for _, wf, _ in devdata]
    
    testdata = [l.strip().split('\t') for l in open(TEST_FN).read().split('\n') if l.strip() != '']
    itestdata = [([c for c in lemma],tags.split(';')) for lemma, tags in testdata]

    characters = set([EOS])
    
    for wf in idata + odata + idevdata + odevdata:
        for c in wf:
            characters.add(c)

    int2char = list(characters)
    char2int = {c:i for i,c in enumerate(characters)}
    VOCAB_SIZE = len(characters)
    
    model = dy.Model()
    
    enc_fwd_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, model)
    enc_bwd_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, model)
    dec_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, STATE_SIZE*2+EMBEDDINGS_SIZE, STATE_SIZE, model)
    
    input_lookup = model.add_lookup_parameters((VOCAB_SIZE, EMBEDDINGS_SIZE))
    attention_w1 = model.add_parameters( (ATTENTION_SIZE, STATE_SIZE*2))
    attention_w2 = model.add_parameters( (ATTENTION_SIZE, STATE_SIZE*LSTM_NUM_OF_LAYERS*2))
    attention_v = model.add_parameters( (1, ATTENTION_SIZE))
    decoder_w = model.add_parameters( (VOCAB_SIZE, STATE_SIZE))
    decoder_b = model.add_parameters( (VOCAB_SIZE))
    output_lookup = model.add_lookup_parameters((VOCAB_SIZE, EMBEDDINGS_SIZE))
    
    train(model, idata,odata,idevdata,odevdata,itestdata,open(O_FN,'w'))
