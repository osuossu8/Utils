
# from https://gist.github.com/jamesr2323/33c67ba5ac29880171b63d2c7f1acdc5
# Thanks https://discuss.pytorch.org/t/rmse-loss-function/16540

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss