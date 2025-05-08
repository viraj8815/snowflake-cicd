CREATE OR REPLACE FUNCTION infer_model_v2(
    ss_sales_price FLOAT,
    ss_quantity FLOAT,
    ss_ext_discount_amt FLOAT,
    ss_net_profit FLOAT,
    d_year INT,
    d_month INT,
    d_day INT,
    s_closed_date_sk INT
)
RETURNS FLOAT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.8'
HANDLER = 'predict'
PACKAGES = ('scikit-learn', 'pandas', 'cloudpickle')
IMPORTS = ('@ml_models_stage/model.pkl.gz')
AS
$$
import cloudpickle
import gzip
import sys
import os

model_path = os.path.join(sys._xoptions["snowflake_import_directory"], "model.pkl.gz")

with gzip.open(model_path, "rb") as f:
    model = cloudpickle.load(f)

def predict(ss_sales_price, ss_quantity, ss_ext_discount_amt, ss_net_profit, d_year, d_month, d_day, s_closed_date_sk):
    features = [[
        ss_sales_price,
        ss_quantity,
        ss_ext_discount_amt,
        ss_net_profit,
        d_year,
        d_month,
        d_day,
        s_closed_date_sk
    ]]
    return float(model.predict(features)[0])
$$;
