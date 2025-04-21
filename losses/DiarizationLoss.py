import torch


class DiarizationLoss(torch.nn.Module):
    def __init__(self, triplet_lambda, bce_lambda):
        super().__init__()
        self.triplet_loss = torch.nn.TripletMarginLoss(margin=0.5, p=2)
        self.bce_loss = torch.nn.BCELoss()
        self.triplet_lambda = triplet_lambda
        self.bce_lambda = bce_lambda

    def forward(self, anchors, positives, negatives, logits, labels):
        triplet_loss = self.triplet_loss(anchors, positives, negatives)
        bce_loss = self.bce_loss(
            logits.reshape(-1).type(torch.FloatTensor),
            labels.reshape(-1).type(torch.FloatTensor),
        )
        return self.triplet_lambda * triplet_loss + self.bce_lambda * bce_loss

class DiarizationLogitsLoss(torch.nn.Module):
   def __init__(self, triplet_lambda, bce_lambda):
      super().__init__()
      self.triplet_loss = torch.nn.TripletMarginLoss(margin = 0.5, p = 2)
      self.bce_loss = torch.nn.BCEWithLogitsLoss()
      self.triplet_lambda = triplet_lambda
      self.bce_lambda = bce_lambda
   def forward(self, anchors, positives, negatives, logits, labels):
      triplet_loss = self.triplet_loss(anchors, positives, negatives)
      logits, labels = logits.reshape(-1, 1), labels.reshape(-1, 1)
      bce_loss = self.bce_loss(logits, labels)
      return self.triplet_lambda * triplet_loss + self.bce_lambda * bce_loss
       
# class DiarizationLoss(torch.nn.Module):
#    def __init__(self, triplet_lambda, cross_entropy_lambda):
#       super().__init__()
#       self.triplet_loss = torch.nn.TripletMarginLoss(margin = 0.5, p = 2)
#       self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
#       self.triplet_lambda = triplet_lambda
#       self.cross_entropy_lambda = cross_entropy_lambda
#    def forward(self, anchors, positives, negatives, logits, labels):
#       triplet_loss = self.triplet_loss(anchors, positives, negatives)
#       cross_entropy_loss = self.cross_entropy_loss(logits, labels)
#       return self.triplet_lambda * triplet_loss + self.cross_entropy_lambda * cross_entropy_loss
