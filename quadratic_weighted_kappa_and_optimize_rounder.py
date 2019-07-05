
from sklearn.metrics import cohen_kappa_score
from logging import getLogger, Formatter, StreamHandler, FileHandler, INFO

'''
回帰で求めたカッパスコアを、閾値で N値に分類する。 (cf: pet finder)

# https://www.kaggle.com/kaerunantoka/ynktk-480-v2-fee-per-pet-stats-k

'''
COMPETITION_NAME = 'petfinder-adoption-prediction'
logger = getLogger(COMPETITION_NAME)
LOGFORMAT = '%(asctime)s %(levelname)s %(message)s'

def init_logger():
    # Add handlers
    handler = StreamHandler()
    handler.setLevel(INFO)
    handler.setFormatter(Formatter(LOGFORMAT))
    fh_handler = FileHandler('{}.log'.format(MODEL_NAME))
    fh_handler.setFormatter(Formatter(LOGFORMAT))
    logger.setLevel(INFO)
    logger.addHandler(handler)
    logger.addHandler(fh_handler)

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    logger.info(f'[{name}] done in {time.time() - t0:.0f} s')



def get_score(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def to_bins(x, borders):
    for i in range(len(borders)):
        if x <= borders[i]:
            return i
    return len(borders)

class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _loss(self, coef, X, y, idx):
        X_p = np.array([to_bins(pred, coef) for pred in X])
        ll = -get_score(y, X_p)
        return ll

    def fit(self, X, y):
        coef = [0.2, 0.4, 0.6, 0.8]
        golden1 = 0.618
        golden2 = 1 - golden1
        ab_start = [(0.01, 0.3), (0.15, 0.56), (0.35, 0.75), (0.6, 0.9)]
        for it1 in range(10):
            for idx in range(4):
                # golden section search
                a, b = ab_start[idx]
                # calc losses
                coef[idx] = a
                la = self._loss(coef, X, y, idx)
                coef[idx] = b
                lb = self._loss(coef, X, y, idx)
                for it in range(20):
                    # choose value
                    if la > lb:
                        a = b - (b - a) * golden1
                        coef[idx] = a
                        la = self._loss(coef, X, y, idx)
                    else:
                        b = b - (b - a) * golden2
                        coef[idx] = b
                        lb = self._loss(coef, X, y, idx)
        self.coef_ = {'x': coef}

    def predict(self, X, coef):
        X_p = np.array([to_bins(pred, coef) for pred in X])
        return X_p

    def coefficients(self):
        return self.coef_['x']



with timer('optimize threshold'):
    optR = OptimizedRounder()
    optR.fit(y_pred, y)
    coefficients = optR.coefficients()
    y_pred = optR.predict(y_pred, coefficients)
    score = get_score(y, y_pred)
    logger.info(f'Coefficients = {coefficients}')
    logger.info(f'QWK = {score}')
    y_test = optR.predict(y_test, coefficients).astype(int)