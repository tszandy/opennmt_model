import torch
import onmt
from onmt.utils.misc import tile
modelpath = '/mnt/nfs/work1/696ds-s19/wxie/wiki_model/wiki_model_step_10000.pt'
# textpath = '/mnt/nfs/work1/696ds-s19/wxie/DATA/example.source.txt'
testpath = '/mnt/nfs/work1/696ds-s19/wxie/arc_data/ARC-Challenge-Train.txt'
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model','-model',dest='models', metavar='MODEL',
              nargs='+', type=str, default=[modelpath])
parser.add_argument('--gpu', '-g', type=int, default=0)
parser.add_argument('--fp32', '-f', type=str, default=False)
parser.add_argument('--src', '-x', type=str, default=textpath)
parser.add_argument('--shard_size', type=int, default=1000000)
parser.add_argument('--tgt','-tgt')
parser.add_argument('--alpha', '-alpha', type=float, default=0., help="Google NMT length penalty parameter " "(higher = longer generation)")
parser.add_argument('--beta', '-beta', type=float, default=-0., help="Coverage penalty parameter")
parser.add_argument('--length_penalty', '-length_penalty', default='none', choices=['none', 'wu', 'avg'], help="Length Penalty to use.")
parser.add_argument('--coverage_penalty', '-coverage_penalty', default='none', choices=['none', 'wu', 'summary'], help="Coverage Penalty to use.")
parser.add_argument('--beam_size', '-beam_size', type=int, default=5, help='Beam size')
parser.add_argument('--min_length', '-min_length', type=int, default=0, help='Minimum prediction length')
parser.add_argument('--max_length', '-max_length', type=int, default=100, help='Maximum prediction length.')

opt = parser.parse_args()


from onmt.utils.misc import split_corpus
from itertools import repeat

src_shards = split_corpus(opt.src, opt.shard_size)
tgt_shards = split_corpus(opt.tgt, opt.shard_size) \
        if opt.tgt is not None else repeat(None)
src = next(src_shards)

from onmt.model_builder import load_test_model
field, model, model_opt = load_test_model(opt)
import onmt.inputters as inputters
data_type='text'
src_reader = inputters.str2reader[data_type].from_opt(opt)
tgt_reader = inputters.str2reader["text"].from_opt(opt)

data = inputters.Dataset(
    field,
    readers=([src_reader, tgt_reader]),
    data=[("src", src)],
    dirs=[None],
    sort_key=inputters.str2sortkey[data_type],
    filter_pred=None
)

batch_size = 10

data_iter = inputters.OrderedIterator(
    dataset=data,
    device=torch.device("cuda", 0),
    batch_size=batch_size,
    train=False,
    sort=False,
    sort_within_batch=True,
    shuffle=False
)

# field['src'][0][1].vocab
from onmt.translate.beam_search import BeamSearch

src_field = field['src'][0][1]
tgt_field = field['tgt'][0][1]
src_vocab = src_field.vocab
tgt_vocab = tgt_field.vocab

tgt_eos_idx = tgt_vocab.stoi[tgt_field.eos_token]
tgt_pad_idx = tgt_vocab.stoi[tgt_field.pad_token]
tgt_bos_idx = tgt_vocab.stoi[tgt_field.init_token]
tgt_unk_idx = tgt_vocab.stoi[tgt_field.unk_token]
tgt_vocab_len = len(tgt_vocab)
n_best=1
beam_size=opt.beam_size
exclusion_idxs=[]
min_length=opt.min_length
ratio=0.
max_length=opt.max_length
mb_device = 0 
stepwise_penalty=None
block_ngram_repeat=0
global_scorer=onmt.translate.GNMTGlobalScorer.from_opt(opt)
return_attention=False



for batch in data_iter:
    print()
    src, src_lengths = batch.src
    memory_lengths = src_lengths
    enc_states, memory_bank, src_lengths = model.encoder(
                src, src_lengths)
    model.decoder.init_state(src, memory_bank, enc_states)
    model.decoder.map_state(
            lambda state, dim: tile(state, beam_size, dim=dim))
    if isinstance(memory_bank, tuple):
        memory_bank = tuple(tile(x, beam_size, dim=1) for x in memory_bank)
        mb_device = memory_bank[0].device
    else:
        memory_bank = tile(memory_bank, beam_size, dim=1)
        mb_device = memory_bank.device
    memory_lengths = tile(src_lengths, beam_size)
    beam = BeamSearch(
        beam_size,
        n_best=n_best,
        batch_size=batch_size,
        global_scorer=global_scorer,
        pad=tgt_pad_idx,
        eos=tgt_eos_idx,
        bos=tgt_bos_idx,
        min_length=min_length,
        ratio=ratio,
        max_length=max_length,
        mb_device=mb_device,
        return_attention=return_attention,
        stepwise_penalty=stepwise_penalty,
        block_ngram_repeat=block_ngram_repeat,
        exclusion_tokens=exclusion_idxs,
        memory_lengths=memory_lengths)
    for step in range(max_length):
        decoder_input = beam.current_predictions.view(1, -1, 1)
        dec_out, dec_attn = model.decoder(decoder_input, memory_bank, memory_lengths=memory_lengths, step=step)
        log_probs = model.generator(dec_out.squeeze(0))
        attn = dec_attn['std']
        log_probs,attn = log_probs.detach(),attn.detach()
        beam.advance(log_probs, attn)
        any_beam_is_finished = beam.is_finished.any()
        if any_beam_is_finished:
            beam.update_finished()
            if beam.done:
                break
        select_indices = beam.current_origin
        if any_beam_is_finished:
            if isinstance(memory_bank, tuple):
                memory_bank = tuple(x.index_select(1, select_indices)
                                        for x in memory_bank)
            else:
                memory_bank = memory_bank.index_select(1, select_indices)
            memory_lengths = memory_lengths.index_select(0, select_indices)
        model.decoder.map_state(
            lambda state, dim: state.index_select(dim, select_indices))
    print(src.shape)
    # print([list(map(lambda x:tgt_vocab.itos[x],input)) for input in src.transpose(1,0)])
    # print(beam.predictions)
    # print([list(map(lambda x:tgt_vocab.itos[x],output[0])) for output in beam.predictions])
    for input in src.transpose(1,0):
        print(list(map(lambda x:src_vocab.itos[x],input)))
    print('---------------------')
    for output in beam.predictions:
        print(list(map(lambda x:tgt_vocab.itos[x],output[0])))

# print(i,(src_shard, tgt_shard))









