import pandas as pd
from src.train.train import build_interactions

def test_loss_decreasing():
    df=pd.DataFrame({'user_id':[1,1,2],'item_id':[10,11,10],'rating':[5,4,3]})
    inter,n_users,n_items,um,im = build_interactions(df)
    from src.model.matrix_factorization import MatrixFactorization
    mf=MatrixFactorization(n_users,n_items)
    loss=mf.train(inter,epochs=3,batch_size=2,verbose=False)
    assert loss[-1] <= loss[0]
