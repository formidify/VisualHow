__author__ = 'Licheng'
from pprint import pprint
# from tokenizer.ptbtokenizer import PTBTokenizer
from vist_eval.bleu.bleu import Bleu
from vist_eval.meteor.meteor import Meteor
from vist_eval.rouge.rouge import Rouge
from vist_eval.cider.cider import Cider
from spice.spice import Spice


class AlbumEvaluator:
    def __init__(self):
        """
        :params vist_sis: vist's Story_in_Sequence instance
        :params preds   : [{'album_id', 'pred_story_str'}]
        """
        self.eval_overall = {}  # overall score
        self.eval_albums = {}  # score on each album_id
        self.album_to_eval = {}  # album_id -> eval

    def evaluate(self, album_to_Gts, album_to_Res):
        self.album_to_Res = album_to_Res
        self.album_to_Gts = album_to_Gts

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = []
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),  # df='VIST/VIST-train-words'
            # (Spice(), "Spice")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score ...' % (scorer.method()))
            score, scores = scorer.compute_score(self.album_to_Gts, self.album_to_Res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setAlbumToEval(scs, self.album_to_Gts.keys(), m)
                    print('%s: %.3f' % (m, sc))
            else:
                self.setEval(score, method)
                self.setAlbumToEval(scores, self.album_to_Gts.keys(), method)
                print('%s: %.3f' % (method, score))

        self.setEvalAlbums()

    def setEval(self, score, method):
        self.eval_overall[method] = score

    def setAlbumToEval(self, scores, album_ids, method):
        for album_id, score in zip(album_ids, scores):
            if not album_id in self.album_to_eval:
                self.album_to_eval[album_id] = {}
                self.album_to_eval[album_id]['album_id'] = album_id
            self.album_to_eval[album_id][method] = score

    def setEvalAlbums(self):
        self.eval_albums = [eval for album_id, eval in self.album_to_eval.items()]
