CREATE OR REPLACE FUNCTION predict_category(
    ss_sales_price FLOAT,
    ss_quantity FLOAT,
    ss_ext_discount_amt FLOAT,
    ss_net_profit FLOAT,
    d_year FLOAT,
    d_month_seq FLOAT,
    d_day FLOAT,
    s_closed_date_sk FLOAT
)
RETURNS FLOAT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.8'
HANDLER = 'predict'
IMPORTS = ('@ml_models_stage/model.pkl.gz')
AS
$$
import gzip
import pickle

with gzip.open("model.pkl.gz", "rb") as f:
    model = pickle.load(f)

def predict(ss_sales_price, ss_quantity, ss_ext_discount_amt, ss_net_profit,
            d_year, d_month_seq, d_day, s_closed_date_sk):
    features = [[
        ss_sales_price,
        ss_quantity,
        ss_ext_discount_amt,
        ss_net_profit,
        d_year,
        d_month_seq,
        d_day,
        s_closed_date_sk
    ]]
    prediction = model.predict(features)
    return float(prediction[0])
$$;
