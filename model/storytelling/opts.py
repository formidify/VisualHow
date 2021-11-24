import argparse
import os

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aggregate_mode', default="att", type=str,
                        help='Mode of aggregation of feature. ["att"]')
    parser.add_argument('--attention_loss', default="ce", type=str,
                        help='Mode of attention loss. ["bce", "ce", None"]')
    parser.add_argument('--attention_loss_weight', default=0.2, type=float,
                        help='Weight of the attention loss')
    parser.add_argument('--backbone_source', type=str, default='imagenet',
                        help='the source of the backbone model, detector|imagenet')
    parser.add_argument('--data_name', default='wikihow',
                        help='{coco,f30k}_precomp')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--no_imgnorm', action='store_false',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--embed_size', default=2048, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--backbone_path', type=str, default='',
                        help='path to the pre-trained backbone net')
    parser.add_argument('--precomp_enc_type', default="backbone",
                        help='basic|backbone')



    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--option', type=str, default='train', help='train | test')
    parser.add_argument('--id', type=str, default='default', help='an id identifying this run/job')
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    # Data input settings
    # parser.add_argument('--data_dir', type=str, default='/mnt/sshd/xwang/VIST')
    parser.add_argument('--data_dir', type=str, default='/media/CVPR_2022/data/wikihow')
    parser.add_argument('--desc_h5', type=str, default='wikiHow/description.h5')
    parser.add_argument('--story_h5', type=str, default='wikiHow/story.h5')
    parser.add_argument('--full_story_h5', type=str, default='wikiHow/full_story.h5')
    parser.add_argument('--title_h5', type=str, default='wikiHow/title.h5')
    parser.add_argument('--story_line_json', type=str, default='wikiHow/story_line.json')
    parser.add_argument('--resume_from', type=str, default=None,
                        help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                        'infos.pkl'         : configuration;
                        'checkpoint'        : paths to model file(s) (created by tf).
                                              Note: this file contains absolute paths, be careful when moving files around;
                        'model.ckpt-*'      : file(s) with model definition (created by tf)""")
    parser.add_argument('--start_from_model', type=str, default=None)
    parser.add_argument('--use_title', type=bool, default=True,
                        help='set to True to use the title for the model ')

    # Model settings
    parser.add_argument('--model', type=str, default="BaseModel")
    parser.add_argument('--task', type=str, default="story_telling", help='story_telling')
    parser.add_argument('--rnn_type', type=str, default='gru', help='gru, lstm')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in the RNN')
    parser.add_argument('--word_embed_dim', type=int, default=512,
                        help='the encoding size of each token in the vocabulary, and the image.')
    parser.add_argument('--feat_size', type=int, default=2048, help='2048 for resnet, 4096 for vgg')
    parser.add_argument('--conv_feat_size', type=int, default=7 * 7 * 2048, help='2048 for resnet, 4096 for vgg')
    parser.add_argument('--use_conv', type=bool, default=False)
    parser.add_argument('--story_size', type=int, default=5,
                        help='number of images/sentences appearing in each story')
    parser.add_argument('--with_position', type=bool, default=False,
                        help='whether to use position embedding for the image feature')
    parser.add_argument('--max_story_size', type=int, default=10,
                        help='maximum number of images/sentences appearing in each story')
    parser.add_argument('--max_title_size', type=int, default=10,
                        help='maximum number of title size appearing in whole story')

    # Optimization: General
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--shuffle', type=bool, default=True,
                        help='set to True to have the data reshuffled at every epoch during training ')
    parser.add_argument('--batch_size', type=int, default=32, help='minibatch size')
    parser.add_argument('--workers', type=int, default=8,
                        help='number of workers to load data')
    parser.add_argument('--grad_clip', type=float, default=10,
                        help='clip gradients at this value')
    parser.add_argument('--visual_dropout', type=float, default=0.2,
                        help='strength of dropout in the Language Model RNN')
    parser.add_argument('--title_dropout', type=float, default=0.2,
                        help='strength of dropout in the Language Model RNN')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='strength of dropout in the Language Model RNN')
    parser.add_argument('--beam_size', type=int, default=1,
                        help='indicates number of beams in beam search. This is only used in the evaluation mode')

    # Optimization: for the Language Model
    parser.add_argument('--optim', type=str, default='Adam',
                        help='RMSprop|SGD|momSGD|Adam|Adadelta|YF')
    parser.add_argument('--learning_rate', type=float, default=4e-4, help='learning rate')
    parser.add_argument('--learning_rate_decay_start', type=int, default=-1,
                        help='from which epoch to start decaying learning rate? (-1 = dont)')
    parser.add_argument('--learning_rate_decay_every', type=int, default=5,
                        help='every how many epochs thereafter to drop LR')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.5,
                        help='every how many epochs thereafter to drop LR')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum, only used in rmsprop & sgd')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
    # Adam optimizer setting
    parser.add_argument('--optim_alpha', type=float, default=0.8, help='alpha for momentum')
    parser.add_argument('--optim_beta', type=float, default=0.999, help='beta for adam')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                        help='epsilon that goes into denominator for smoothing')

    # Reinforcement learning
    parser.add_argument('--start_rl', type=int, default=-1,
                        help='at what epoch to start RL training, -1 means never')
    parser.add_argument('--reward_type', type=str, default='METEOR',
                        help="CIDEr | ROUGE_L | METEOR | Bleu_4 | Bleu_3")
    parser.add_argument('--rl_weight', type=float, default=0.5)
    parser.add_argument('--cached_tokens', type=str, default='wikiHow/wikiHow-train-words',
                        help='used to compute CIDEr reward')
    parser.add_argument('--use_feature_with_context', type=bool, default=False)
    parser.add_argument('--decoding_method_DISC', type=str, default='sample', help='greedy | sample')

    # Schedule sampling
    parser.add_argument('--scheduled_sampling_start', type=int, default=0,
                        help='at what epoch to start decay gt probability, -1 means never')
    parser.add_argument('--scheduled_sampling_increase_every', type=int, default=5,
                        help='every how many epochs to increase scheduled sampling probability')
    parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05,
                        help='How much to update the prob')
    parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25,
                        help='Maximum scheduled sampling prob.')

    # Evaluation/Checkpointing
    parser.add_argument('--metric', type=str, default='CIDEr',
                        help="XE | CIDEr | ROUGE_L | METEOR | Bleu_4 | Bleu_3")
    parser.add_argument('--save_checkpoint_every', type=int, default=1000,
                        help='how often to save a model checkpoint (in iterations)')
    parser.add_argument('--checkpoint_path', type=str, default='data/save',
                        help='directory to store checkpointed models')
    parser.add_argument('--losses_log_every', type=int, default=10,
                        help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
    parser.add_argument('--load_best_score', type=bool, default=True,
                        help='whether to load previous best score when resuming training.')
    parser.add_argument('--prefix', type=str, default='fc/', help="feature folder")

    # misc
    parser.add_argument('--always', type=str, default=None, help='always train one model, no alternating training')
    parser.add_argument('--D_iter', type=int, default=100, help='Discriminator update iterations')
    parser.add_argument('--G_iter', type=int, default=100, help='Generator update iterations')
    parser.add_argument('--activation', type=str, default="sign",
                        help='the last activation function of the reward model: sign | tahn')

    args = parser.parse_args()

    if args.start_rl >= 0:
        args.metric = args.reward_type

    return args
