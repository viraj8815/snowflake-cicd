CREATE OR REPLACE FUNCTION PREDICT_CATEGORY(
    ss_sales_price FLOAT,
    ss_quantity FLOAT,
    ss_ext_discount_amt FLOAT,
    ss_net_profit FLOAT
)
RETURNS STRING
LANGUAGE PYTHON
RUNTIME_VERSION = 3.8
HANDLER = 'predict'
IMPORTS = ('@ml_models_stage/model.pkl.gz')
AS
$$
import _pickle as pickle
import gzip

def predict(ss_sales_price, ss_quantity, ss_ext_discount_amt, ss_net_profit):
    with gzip.open("model.pkl.gz", "rb") as f:
        model = pickle.load(f)
    
    features = [[
        ss_sales_price,
        ss_quantity,
        ss_ext_discount_amt,
        ss_net_profit
    ]]
    return str(model.predict(features)[0])
$$;
