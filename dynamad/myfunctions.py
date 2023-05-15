import pandas as pd
import numpy as np
import tqdm

def conf_region_e1(pairs, e1_sim, k, id1='entity_1'):
       
    c_conf_reg = []
    c_k_ls = []
    e1 = list(set(pairs[0]))

    for i in range(len(e1)):
        # Top ranking compounds in the training space
        c_k_sims = e1_sim.loc[:, e1_sim.columns == e1[i]].sort_values(by=e1[i], ascending=False)[0:k]
        c_k_ids = e1_sim[e1_sim.index.isin(e1_sim.loc[:, e1_sim.columns == e1[i]].sort_values(by=e1[i], ascending=False)[0:k].index)][id1]
        c_k = pd.merge(c_k_ids, c_k_sims, left_index=True, right_index=True)

        c_conf_reg.append(c_k_ids)
        c_k_ls.append(c_k)

    # Append c_k_ids to c_conf_reg as a dictionary
    c_conf_reg = dict(zip(e1, c_conf_reg))
    c_k_ls = dict(zip(e1, c_k_ls))

    return c_conf_reg, c_k_ls


def conf_region_e2(pairs, e2_sim, q, id2='entity_2'):
            
    t_conf_reg = []
    t_q_ls = []
    e2 = list(set(pairs[1]))

    for i in range(len(e2)):
        # Top ranking targets in the training space
        t_q_sims = e2_sim.loc[:, e2_sim.columns == e2[i]].sort_values(by=e2[i], ascending=False)[0:q]
        t_q_ids = e2_sim[e2_sim.index.isin(e2_sim.loc[:, e2_sim.columns == e2[i]].sort_values(by=e2[i], ascending=False)[0:q].index)][id2]
        t_q = pd.merge(t_q_ids, t_q_sims, left_index=True, right_index=True)

        t_conf_reg.append(t_q_ids)
        t_q_ls.append(t_q)

    # Append t_q_ids to t_conf_reg as a dictionary
    t_conf_reg = dict(zip(e2, t_conf_reg))
    t_q_ls = dict(zip(e2, t_q_ls))

    return t_conf_reg, t_q_ls


def interacting_pairs(pairs, ispace, c_conf, t_conf, i):
    
    # Find the interaction landscape of extracted conformity subspaces
    conf_region = pd.merge(c_conf[pairs.loc[i, 'entity_1']], ispace)
    conf_region = conf_region.loc[:, conf_region.columns.isin(t_conf[pairs.loc[i, 'entity_2']])]
    conf_region.insert(0, "entity_1", c_conf[pairs.loc[i, 'entity_1']].values)
    conf_region = pd.melt(conf_region, id_vars=['entity_1']).rename(columns={"variable": "entity_2", "value": "y"}).dropna()

    return conf_region


# I) Dynamic applicability domain (CV variant)
def dad_cv(x, pairs, tr_pairs, cv, tr_y, model, e1_sim, e2_sim, ispace, k, q, id1='entity_1', id2='entity_2',  type_m=['x_conf', 'cal_conf']):
    
    # Non-conformity set
    pairs_df = pd.DataFrame(pairs).T
    pairs_df.columns = ['entity_1', 'entity_2']
    pairs_df['y_pred'] = model.predict(x)
    ts_conf_sizes = []

    # Assign CV predictions for training samples
    tr_pairs_df = pd.DataFrame(tr_pairs).T
    tr_pairs_df.columns = ['entity_1', 'entity_2']
    tr_pairs_df['y'] = tr_y
    tr_pairs_df['cv_pred'] = cv

    # Confidence ranges
    conf=['0.75', '0.80', '0.85', '0.90', '0.95', '0.99']

    # Find the conformity regions for both entities
    c_conf_reg, c_k_ls = conf_region_e1(pairs=pairs, e1_sim=e1_sim, k=k, id1=id1)
    t_conf_reg, t_q_ls = conf_region_e2(pairs=pairs, e2_sim=e2_sim, q=q, id2=id2)
    
    for i in tqdm.tqdm(range(0, len(pairs_df))):
        try:

            # Find the interaction landscape of extracted conformity subspaces
            conf_region = interacting_pairs(pairs=pairs_df, ispace=ispace, c_conf=c_conf_reg, t_conf=t_conf_reg, i=i)
            conf_region = pd.merge(conf_region, tr_pairs_df, on=["entity_1", "entity_2", "y"])
            conf_region['nonconf'] = abs(conf_region['y'] - conf_region['cv_pred'])
            conf_region['pred_diff'] = abs(conf_region['y'] - pairs_df['y_pred'][i])
            conf_region = conf_region.drop('cv_pred', axis=1)

            # Unite similarities and differences under one df
            ts_nonconf = conf_region.merge(c_k_ls[pairs_df.loc[i, 'entity_1']], on="entity_1")
            ts_nonconf = ts_nonconf.merge(t_q_ls[pairs_df.loc[i, 'entity_2']], on="entity_2")

            # Check conformity region sizes!
            ts_conf_sizes.append(len(ts_nonconf))

            if type_m == 'cal_conf':
                # Compute alpha scores - classic cp logic
                for j in range(len(ts_nonconf)):
                    ts_nonconf.loc[j,'alpha_sc'] = (ts_nonconf[ts_nonconf['nonconf'] <= ts_nonconf.loc[j,'nonconf']].shape[0]) / ts_nonconf.shape[0]

            else:
                # Compute alpha scores - dAD logic
                for j in range(len(ts_nonconf)):
                    ts_nonconf.loc[j,'alpha_sc'] = (ts_nonconf[ts_nonconf['pred_diff'] <= ts_nonconf.loc[j,'nonconf']].shape[0]) / ts_nonconf.shape[0]

            
            # Choose alfa for wanted confidence                 
            for c in conf:
                if c == '0.99':
                    pairs_df.loc[i,c] = np.min(ts_nonconf[ts_nonconf['alpha_sc'].between(float(c), 1)].nonconf)
                else:
                    pairs_df.loc[i,c] = np.min(ts_nonconf[ts_nonconf['alpha_sc'].between(float(c), float(c)+0.04)].nonconf)
        except:
            continue

    return pairs_df, ts_conf_sizes




# II) Dynamic applicability domain (NN variant)
def dad_nn(x, pairs, model, e1_sim, e2_sim, ispace, k, q, id1='entity_1', id2='entity_2',  type_m=['x_conf', 'cal_conf']):
    
    # Non-conformity set
    pairs_df = pd.DataFrame(pairs).T
    pairs_df.columns = ['entity_1', 'entity_2']
    pairs_df['y_pred'] = model.predict(x)
    ts_conf_sizes = []

    # Confidence ranges
    conf=['0.75', '0.80', '0.85', '0.90', '0.95', '0.99']

    # Find the conformity regions for both entities
    c_conf_reg, c_k_ls = conf_region_e1(pairs=pairs, e1_sim=e1_sim, k=k, id1=id1)
    t_conf_reg, t_q_ls = conf_region_e2(pairs=pairs, e2_sim=e2_sim, q=q, id2=id2)
    
    for i in tqdm.tqdm(range(0,len(pairs_df))):
        try:

            # Find the interaction landscape of extracted conformity subspaces
            conf_region = interacting_pairs(pairs=pairs_df, ispace=ispace, c_conf=c_conf_reg, t_conf=t_conf_reg, i=i)
            
            conf_region['nonconf'] = abs(conf_region['y'] - conf_region['y'].mean())
            conf_region['pred_diff'] = abs(conf_region['y'] - pairs_df['y_pred'][i])

            # Unite similarities and differences under one df
            ts_nonconf = conf_region.merge(c_k_ls[pairs_df.loc[i, 'entity_1']], on="entity_1")
            ts_nonconf = ts_nonconf.merge(t_q_ls[pairs_df.loc[i, 'entity_2']], on="entity_2")

            # Check conformity region sizes!
            ts_conf_sizes.append(len(ts_nonconf))

            if type_m == 'cal_conf':
                # Compute alpha scores - classic cp logic
                for j in range(len(ts_nonconf)):
                    ts_nonconf.loc[j,'alpha_sc'] = (ts_nonconf[ts_nonconf['nonconf'] <= ts_nonconf.loc[j,'nonconf']].shape[0]) / ts_nonconf.shape[0]

            else:
                # Compute alpha scores - dAD logic
                for j in range(len(ts_nonconf)):
                    ts_nonconf.loc[j,'alpha_sc'] = (ts_nonconf[ts_nonconf['pred_diff'] <= ts_nonconf.loc[j,'nonconf']].shape[0]) / ts_nonconf.shape[0]

            
            # Choose alfa for wanted confidence                 
            for c in conf:
                if c == '0.99':
                    pairs_df.loc[i,c] = np.min(ts_nonconf[ts_nonconf['alpha_sc'].between(float(c), 1)].nonconf)
                else:
                    pairs_df.loc[i,c] = np.min(ts_nonconf[ts_nonconf['alpha_sc'].between(float(c), float(c)+0.04)].nonconf)
        except:
            continue

    return pairs_df, ts_conf_sizes

