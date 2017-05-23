from __future__ import print_function
from sys import argv, stderr
import random
import pickle

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

BEAM=5
from math import log

def beam_generate(in_seq, enc_fwd_lstm, enc_bwd_lstm, dec_lstm):
    embedded = embed_sentence(in_seq)
    encoded = encode_sentence(enc_fwd_lstm, enc_bwd_lstm, embedded)

    w = dy.parameter(decoder_w)
    b = dy.parameter(decoder_b)
    w1 = dy.parameter(attention_w1)
    input_mat = dy.concatenate_cols(encoded)
    w1dt = None

    histories = [[0.0,output_lookup[char2int[EOS]],'',None]]

    s = dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(STATE_SIZE * 2), histories[0][1]]))
    histories[0][3] = s

    count_EOS = 0
    for i in range(len(in_seq)*2):
        if count_EOS == 2: break
        # w1dt can be computed and cached once for the entire decoding phase
        presents = []
        for ll, embedding, out, s in histories:
            w1dt = w1dt or w1 * input_mat        
            vector = dy.concatenate([attend(input_mat, s, w1dt), embedding])
            new_s = s.add_input(vector)
            out_vector = w * new_s.output() + b
            probs = dy.softmax(out_vector).vec_value()            
            probs = sorted([(prob,j) for j, prob in enumerate(probs)], key = lambda x:x[0], reverse=1)
            for prob, j in probs:      
                next_embedding = output_lookup[j]                
                presents.append([ll + log(prob), next_embedding, out + int2char[j], new_s])

        presents.sort(reverse=1,key=lambda x:x[0])
        histories = presents[:BEAM]
        if presents[0][2].endswith(EOS) and presents[0][2] != EOS:
            return presents[0][2].replace(EOS,'')

    return histories[0][2].replace(EOS,'')


def get_loss(input_sentence, output_sentence, enc_fwd_lstm, enc_bwd_lstm, dec_lstm):
    dy.renew_cg()
    embedded = embed_sentence(input_sentence)
    encoded = encode_sentence(enc_fwd_lstm, enc_bwd_lstm, embedded)
    return decode(dec_lstm, encoded, output_sentence)


def train(isentences,osentences, idevsentences,odevsentences, alpha, beta, ofilen):
    trainer = dy.SimpleSGDTrainer(model,e0=alpha,edecay=beta)
    iopairs = list(zip(isentences,osentences))
    random.shuffle(iopairs)
    best_dev_acc = 0
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
        dev_acc = corr * 100.0 / len(idevsentences)

        print("EPOCH %u: LOSS %.2f, DEV ACC %.2f" % (i+1, loss_value/len(iopairs), dev_acc))
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            print("SAVING!")
            save_model(ofilen)

    
def test(itestsentences, ofile):
    for ilemma,ilabels in itestsentences:
        dy.renew_cg()
        sys_o = generate(ilemma+ilabels, enc_fwd_lstm, enc_bwd_lstm, dec_lstm)
        ofile.write("%s\t%s\t%s\n" % (''.join(ilemma),sys_o,';'.join(ilabels)))

def init_models(chars, fn=None):
    global characters, int2char, char2int, model, enc_fwd_lstm, enc_bwd_lstm, \
        dec_lstm, input_lookup, attention_w1, attention_w2, attention_v, \
        decoder_w, decoder_b, output_lookup
    
    characters = None
    int2char = None
    char2int = None

    if fn:
        characters, int2char, char2int = \
            pickle.load(open('%s.chars.pkl' % fn,'rb'))
    else:
        characters = chars
        int2char = sorted(list(characters))
        char2int = {c:i for i,c in enumerate(int2char)}

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

def save_model(ofilen):
    model.save(ofilen)    
    pickle.dump((characters,int2char,char2int),open("%s.chars.pkl" % ofilen,"wb"))

def load_model(ifilen):
    global model
    model.load(ifilen)

    
