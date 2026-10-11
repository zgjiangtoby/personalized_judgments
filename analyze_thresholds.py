"""Fit population and personal ordinal thresholds on crossed person/text splits."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
from multiple_testing import holm

RIDGES = (.01,.1,1.)
PRECISIONS = (.5,2.,8.)
SEED = 20261009


def ordinal(eta,center,raw_gap,y=None):
    gap = np.logaddexp(0,raw_gap)+.01
    a,b = center-gap/2-eta,center+gap/2-eta
    f1,f2 = expit(a),expit(b)
    logp = np.stack([-np.logaddexp(0,-a),
        -np.logaddexp(0,-b)-np.logaddexp(0,a)+np.log(-np.expm1(-gap)),
        -np.logaddexp(0,b)],axis=-1)
    assert np.isfinite(logp).all()
    if y is None:
        return np.exp(logp)
    loss = -logp[np.arange(len(y)),y]
    inverse_gap = 1/np.expm1(np.minimum(gap,700))
    dt1 = np.where(y==0,f1-1,np.where(y==1,f1+inverse_gap,0.))
    dt2 = np.where(y==2,f2,np.where(y==1,f2-1-inverse_gap,0.))
    dc = dt1+dt2
    dg = (dt2-dt1)/2*expit(raw_gap)
    return loss,-dc,dc,dg


def rater_weights(frame,multiplicity=None):
    counts = frame.groupby("rater_id").size()
    w = 1/frame.rater_id.map(counts).to_numpy(dtype=float)
    if multiplicity is not None:
        w *= frame.rater_id.map(multiplicity).fillna(0).to_numpy(dtype=float)
        expanded = frame.rater_id.map(multiplicity).fillna(0).sum()
    else:
        expanded = len(frame)
    assert w.sum()>0
    return w/w.sum()*expanded


def optimize(objective,initial,label,tolerance=1e-5):
    fitted = minimize(objective,initial,jac=True,method="L-BFGS-B",
                      options={"maxiter":2000,"ftol":1e-13,"gtol":1e-9,"maxls":50})
    value,gradient = objective(fitted.x)
    assert fitted.success and np.abs(gradient).max()<tolerance,(label,str(fitted.message),np.abs(gradient).max())
    return fitted.x,{"fit":label,"iterations":int(fitted.nit),"objective":float(value),
                     "max_gradient":float(np.abs(gradient).max()),"message":str(fitted.message)}


def fit_shared(x,frame,ridge,multiplicity=None,initial=None):
    y = frame.hatespeech.to_numpy(dtype=int)
    weights = rater_weights(frame,multiplicity)
    mass = weights.sum()
    p = x.shape[1]
    if initial is None:
        cumulative = np.clip(np.array([np.average(y==0,weights=weights),np.average(y<=1,weights=weights)]),.01,.99)
        thresholds = np.log(cumulative/(1-cumulative))
        gap = max(thresholds[1]-thresholds[0]-.01,.01)
        initial = np.r_[np.zeros(p),thresholds.mean(),np.log(np.expm1(gap))]
    def objective(parameters):
        beta,c,g = parameters[:p],parameters[p],parameters[p+1]
        eta = np.asarray(x@beta).reshape(-1)
        loss,de,dc,dg = ordinal(eta,np.full(len(y),c),np.full(len(y),g),y)
        gradient = np.r_[np.asarray(x.T@(weights*de)).reshape(-1)+ridge*beta,weights@dc,weights@dg]/mass
        return float((weights@loss+ridge/2*(beta@beta))/mass),gradient
    parameters,record = optimize(objective,initial,"shared")
    return {"beta":parameters[:p],"center":parameters[p],"raw_gap":parameters[p+1],
            "parameters":parameters,"ridge":ridge,"record":record}


def group_codes(frame):
    return frame.reconstructed_gender.map({"women":0,"men":1}).fillna(2).to_numpy(dtype=int)


def fit_group(x,frame,shared,multiplicity=None,initial=None):
    y = frame.hatespeech.to_numpy(dtype=int)
    groups = group_codes(frame)
    weights = rater_weights(frame,multiplicity)
    mass = weights.sum()
    eta = np.asarray(x@shared["beta"]).reshape(-1)
    size = 4
    initial = np.zeros(size) if initial is None else initial
    def objective(parameters):
        offsets = parameters[:4].reshape(2,2)
        center = np.full(len(y),shared["center"])
        raw_gap = np.full(len(y),shared["raw_gap"])
        score = eta.copy()
        for group in (0,1):
            selected = groups==group
            center[selected] += offsets[group,0]
            raw_gap[selected] += offsets[group,1]
        loss,de,dc,dg = ordinal(score,center,raw_gap,y)
        gradient = np.empty(size)
        penalty = .01/2*(parameters[:4]@parameters[:4])
        for group in (0,1):
            selected = groups==group
            gradient[2*group] = weights[selected]@dc[selected]+.01*offsets[group,0]
            gradient[2*group+1] = weights[selected]@dg[selected]+.01*offsets[group,1]
        return float((weights@loss+penalty)/mass),gradient/mass
    parameters,record = optimize(objective,initial,"group_thresholds")
    return {"parameters":parameters,"dimensions":0,"ridge":None,"record":record}


def fit_rank_head(x,frame,shared,interaction_ridge=None,multiplicity=None,initial=None):
    """Nested common ranking versus common ranking plus a centered group contrast."""
    y = frame.hatespeech.to_numpy(dtype=int)
    groups = group_codes(frame)
    coding = np.choose(groups,[-.5,.5,0.])
    weights = rater_weights(frame,multiplicity)
    mass = weights.sum()
    p = x.shape[1]
    has_interaction = interaction_ridge is not None
    size = 4+p*(2 if has_interaction else 1)
    initial = np.r_[np.zeros(4),shared["beta"],np.zeros(p) if has_interaction else []] if initial is None else initial
    def objective(parameters):
        offsets = parameters[:4].reshape(2,2)
        beta = parameters[4:4+p]
        eta = np.asarray(x@beta).reshape(-1)
        center = np.full(len(y),shared["center"])
        gap = np.full(len(y),shared["raw_gap"])
        if has_interaction:
            eta += coding*np.asarray(x@parameters[4+p:]).reshape(-1)
        for code in (0,1):
            selected = groups==code
            center[selected] += offsets[code,0]
            gap[selected] += offsets[code,1]
        loss,de,dc,dg = ordinal(eta,center,gap,y)
        gradient = np.empty(size)
        for code in (0,1):
            selected = groups==code
            gradient[2*code] = weights[selected]@dc[selected]+.01*offsets[code,0]
            gradient[2*code+1] = weights[selected]@dg[selected]+.01*offsets[code,1]
        gradient[4:4+p] = np.asarray(x.T@(weights*de)).reshape(-1)+shared["ridge"]*beta
        penalty = .005*(parameters[:4]@parameters[:4])+shared["ridge"]/2*(beta@beta)
        if has_interaction:
            contrast_beta = parameters[4+p:]
            gradient[4+p:] = np.asarray(x.T@(weights*de*coding)).reshape(-1)+interaction_ridge*contrast_beta
            penalty += interaction_ridge/2*(contrast_beta@contrast_beta)
        return float((weights@loss+penalty)/mass),gradient/mass
    parameters,record = optimize(objective,initial,"rank_interaction" if has_interaction else "common_rank_refit")
    return {"parameters":parameters,"dimensions":p,"ridge":interaction_ridge,"record":record,
            "kind":"rank","has_interaction":has_interaction}


def population_predictions(x,frame,shared,group=None):
    eta = np.asarray(x@shared["beta"]).reshape(-1)
    center = np.full(len(frame),shared["center"])
    raw_gap = np.full(len(frame),shared["raw_gap"])
    if group is not None:
        codes = group_codes(frame)
        offsets = group["parameters"][:4].reshape(2,2)
        p = group["dimensions"]
        if group.get("kind")=="rank":
            eta = np.asarray(x@group["parameters"][4:4+p]).reshape(-1)
            if group["has_interaction"]:
                eta += np.choose(codes,[-.5,.5,0.])*np.asarray(x@group["parameters"][4+p:]).reshape(-1)
        for code in (0,1):
            selected = codes==code
            center[selected] += offsets[code,0]
            raw_gap[selected] += offsets[code,1]
            if p and group.get("kind")!="rank":
                eta[selected] += np.asarray(x[selected]@group["parameters"][4+code*p:4+(code+1)*p]).reshape(-1)
    return eta,center,raw_gap


def personalize(x,frame,shared,group,precision):
    # This frame contains selected observed anchor responses only.
    assert frame.calibration.all() and frame.text_split.eq("anchor").all()
    people,indices = np.unique(frame.rater_id.to_numpy(),return_inverse=True)
    eta,center,gap = population_predictions(x,frame,shared,group)
    y = frame.hatespeech.to_numpy(dtype=int)
    size = len(people)
    def objective(parameters):
        offsets = parameters.reshape(size,2)
        loss,_,dc,dg = ordinal(eta,center+offsets[indices,0],gap+offsets[indices,1],y)
        gradient = np.stack([np.bincount(indices,dc,minlength=size),np.bincount(indices,dg,minlength=size)],axis=1)
        gradient += precision*offsets
        return float((loss.sum()+precision/2*(parameters@parameters))/size),gradient.reshape(-1)/size
    parameters,record = optimize(objective,np.zeros(2*size),"two_anchor_personal_MAP",tolerance=1e-8)
    _,gradient = objective(parameters)
    record["max_per_person_gradient"] = float(np.abs(gradient).max()*size)
    assert record["max_per_person_gradient"]<1e-4,record
    return dict(zip(people,parameters.reshape(size,2))),record


def probabilities(x,frame,shared,group=None,personal=None):
    eta,center,gap = population_predictions(x,frame,shared,group)
    if personal is not None:
        offsets = np.asarray([personal.get(rater,np.zeros(2)) for rater in frame.rater_id])
        center += offsets[:,0]
        gap += offsets[:,1]
    p = ordinal(eta,center,gap)
    assert np.isfinite(p).all() and np.allclose(p.sum(1),1)
    return p


def losses(frame,p):
    y = frame.hatespeech.to_numpy(dtype=int)
    return -np.log(np.maximum(p[np.arange(len(y)),y],1e-300))


def mean_loss(frame,p):
    return float(np.average(losses(frame,p),weights=rater_weights(frame)))


def paired_uncertainty(frame,difference,draws=2000):
    weights = rater_weights(frame)
    weights /= weights.sum()
    estimate = float(weights@difference)
    raters,ri = np.unique(frame.rater_id.to_numpy(),return_inverse=True)
    texts,ti = np.unique(frame.item_id.to_numpy(),return_inverse=True)
    centered = weights*(difference-estimate)
    def variance(index):
        count = index.max()+1
        return count/(count-1)*np.square(np.bincount(index,weights=centered)).sum()
    rater_variance,text_variance = variance(ri),variance(ti)
    cell_variance = len(frame)/(len(frame)-1)*np.square(centered).sum()
    raw = rater_variance+text_variance-cell_variance
    sandwich = np.sqrt(max(0.,raw))
    rng = np.random.default_rng(SEED)
    bootstrap = []
    for _ in range(draws):
        rw = rng.multinomial(len(raters),np.full(len(raters),1/len(raters)))
        tw = rng.multinomial(len(texts),np.full(len(texts),1/len(texts)))
        w = weights*rw[ri]*tw[ti]
        assert w.sum()>0
        bootstrap.append(float(np.average(difference,weights=w)))
    pigeonhole = float(np.std(bootstrap,ddof=1))
    se = max(float(sandwich),pigeonhole)
    return {"delta_nll":estimate,"two_way_sandwich_se":float(sandwich),"two_way_raw_variance":float(raw),
            "rater_variance":float(rater_variance),"text_variance":float(text_variance),"cell_variance":float(cell_variance),
            "pigeonhole_se":pigeonhole,"pigeonhole_percentile_interval":np.quantile(bootstrap,[.025,.975]).tolist(),
            "primary_se_maximum":se,"primary_normal_95_interval":[estimate-1.96*se,estimate+1.96*se],
            "normal_approximation_p":float(2*norm.sf(abs(estimate)/se)) if se>0 else None,
            "rater_clusters":len(raters),"text_clusters":len(texts),"draws":draws,
            "weights":"original rater-equal row weights fixed, multiplied by independent rater/text multiplicities, renormalized",
            "conditional_on_fitted_model":True}


def prepare_features(path,items,texts,train_item_ids):
    indices = {str(item):i for i,item in enumerate(items.item_id)}
    metadata = [json.loads(line) for line in path.with_suffix(".items.jsonl").open()]
    assert len(metadata)==len(indices) and len({r["id"] for r in metadata})==len(metadata)
    vectors = np.load(path,mmap_mode="r")
    assert len(vectors)==len(metadata) and np.isfinite(vectors).all()
    vector_index = {row["id"]:i for i,row in enumerate(metadata)}
    assert set(vector_index)==set(indices)
    ordered = np.asarray(vectors[[vector_index[str(item)] for item in items.item_id]],dtype=np.float64)
    training = items.item_id.isin(train_item_ids).to_numpy()
    pca = PCA(n_components=min(128,training.sum()-1,ordered.shape[1]),svd_solver="randomized",random_state=SEED)
    pca.fit(ordered[training])
    reduced = pca.transform(ordered)
    scaler = StandardScaler().fit(reduced[training])
    x = scaler.transform(reduced)
    tfidf = TfidfVectorizer(ngram_range=(1,2),min_df=3,max_features=50000,sublinear_tf=True,dtype=np.float64)
    training_texts = [texts[str(item)] for item in items.loc[training,"item_id"]]
    tfidf.fit(training_texts)
    lexical = tfidf.transform([texts[str(item)] for item in items.item_id])
    return x,lexical,indices,{"PCA_components":x.shape[1],"PCA_explained_variance":float(pca.explained_variance_ratio_.sum()),
        "PCA_fit_unique_texts":int(training.sum()),"TFIDF_vocabulary":len(tfidf.vocabulary_),
        "TFIDF_ngrams":[1,2],"TFIDF_min_df":3,"TFIDF_max_features":50000,"fit_scope":"train nonanchor and anchor items observed by training raters"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings",type=Path,required=True)
    parser.add_argument("--model-name",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--training-refits",type=int,default=100)
    args = parser.parse_args()
    assert not args.output.exists()
    responses = pd.read_parquet(HERE/"work/split/responses.parquet")
    items = pd.read_csv(HERE/"work/split/items.csv")
    texts = {r["id"]:r["text"] for r in map(json.loads,(HERE/"work/split/texts.jsonl").open())}
    train = responses[responses.rater_split.eq("train") & responses.text_split.isin(["train","anchor"])].reset_index(drop=True)
    validation_all = responses[responses.rater_split.eq("validation") & responses.text_split.eq("validation")].reset_index(drop=True)
    test_all = responses[responses.rater_split.eq("test") & responses.text_split.eq("test")].reset_index(drop=True)
    validation = validation_all[validation_all.reconstructed_gender.isin(["women","men"])].reset_index(drop=True)
    test = test_all[test_all.reconstructed_gender.isin(["women","men"])].reset_index(drop=True)
    anchors = {split:responses[responses.rater_split.eq(split)&responses.calibration].reset_index(drop=True) for split in ("validation","test")}
    x,lexical,indices,feature_record = prepare_features(args.embeddings,items,texts,set(train.item_id))
    def matrix(array,frame):
        return array[[indices[str(item)] for item in frame.item_id]]
    xt,xv,xe = [matrix(x,frame) for frame in (train,validation,test)]
    av,ae = [matrix(x,anchors[split]) for split in ("validation","test")]
    fits,tuning = [],[]
    shared_candidates,lexical_candidates = {},{}
    for ridge in RIDGES:
        shared = fit_shared(xt,train,ridge)
        shared_candidates[ridge] = shared
        score = mean_loss(validation,probabilities(xv,validation,shared))
        tuning.append({"stage":"text","ridge":ridge,"validation_nll":score})
        fits.append({"stage":"text","ridge":ridge,**shared["record"]})
        lexical_model = fit_shared(matrix(lexical,train),train,ridge)
        lexical_candidates[ridge] = lexical_model
        score = mean_loss(validation,probabilities(matrix(lexical,validation),validation,lexical_model))
        tuning.append({"stage":"TFIDF","ridge":ridge,"validation_nll":score})
        fits.append({"stage":"TFIDF","ridge":ridge,**lexical_model["record"]})
    selected_ridge = min((r for r in tuning if r["stage"]=="text"),key=lambda r:r["validation_nll"])["ridge"]
    selected_lexical = min((r for r in tuning if r["stage"]=="TFIDF"),key=lambda r:r["validation_nll"])["ridge"]
    shared = shared_candidates[selected_ridge]
    lexical_model = lexical_candidates[selected_lexical]
    group = fit_group(xt,train,shared)
    fits.append({"stage":"group",**group["record"]})
    common_rank = fit_rank_head(xt,train,shared)
    fits.append({"stage":"common_rank",**common_rank["record"]})
    for precision in PRECISIONS:
        personal,record = personalize(av,anchors["validation"],shared,None,precision)
        score = mean_loss(validation,probabilities(xv,validation,shared,personal=personal))
        tuning.append({"stage":"personal","precision":precision,"validation_nll":score})
        fits.append({"stage":"validation_personal","precision":precision,**record})
    precision = min((r for r in tuning if r["stage"]=="personal"),key=lambda r:r["validation_nll"])["precision"]
    interactions = {}
    for ridge in RIDGES:
        model = fit_rank_head(xt,train,shared,ridge)
        interactions[ridge] = model
        personal,record = personalize(av,anchors["validation"],shared,model,precision)
        score = mean_loss(validation,probabilities(xv,validation,shared,model,personal))
        tuning.append({"stage":"interaction","ridge":ridge,"precision":precision,"validation_nll":score})
        fits.extend([{ "stage":"interaction","ridge":ridge,**model["record"]},
                     {"stage":"validation_interaction_personal","ridge":ridge,**record}])
    interaction_ridge = min((r for r in tuning if r["stage"]=="interaction"),key=lambda r:r["validation_nll"])["ridge"]
    interaction = interactions[interaction_ridge]
    # Freeze all tuning choices before computing test losses.
    args.output.mkdir(parents=True)
    freeze = {"text_ridge":selected_ridge,"TFIDF_ridge":selected_lexical,"personal_precision":precision,
              "interaction_ridge":interaction_ridge,"test_losses_opened":False,"tuning":tuning,
              "features":feature_record,"training_refits":args.training_refits}
    (args.output/"validation_freeze.json").write_text(json.dumps(freeze,indent=2)+"\n")
    personal,record = personalize(ae,anchors["test"],shared,None,precision)
    fits.append({"stage":"test_personal_calibration",**record})
    group_personal,record = personalize(ae,anchors["test"],shared,group,precision)
    fits.append({"stage":"test_group_personal_calibration",**record})
    common_rank_personal,record = personalize(ae,anchors["test"],shared,common_rank,precision)
    fits.append({"stage":"test_common_rank_personal_calibration",**record})
    interaction_personal,record = personalize(ae,anchors["test"],shared,interaction,precision)
    fits.append({"stage":"test_interaction_personal_calibration",**record})
    models = {"pooled":(None,None),"group":(group,None),"personal":(None,personal),
              "group_personal":(group,group_personal),"common_rank_personal":(common_rank,common_rank_personal),
              "interaction_personal":(interaction,interaction_personal)}
    all_probabilities = {name:probabilities(matrix(x,test_all),test_all,shared,g,p) for name,(g,p) in models.items()}
    all_probabilities["TFIDF"] = probabilities(matrix(lexical,test_all),test_all,lexical_model)
    primary_mask = test_all.reconstructed_gender.isin(["women","men"]).to_numpy()
    predicted = {name:p[primary_mask] for name,p in all_probabilities.items()}
    nll = {name:losses(test,p) for name,p in predicted.items()}
    metrics = {}
    for name,p in predicted.items():
        y = test.hatespeech.to_numpy(dtype=int)
        w = rater_weights(test)
        metrics[name] = {"nll":mean_loss(test,p),"brier":float(np.average(((p-np.eye(3)[y])**2).sum(1),weights=w)),
                         "accuracy":float(np.average(p.argmax(1)==y,weights=w))}
    pairs = [("pooled","TFIDF"),("personal","pooled"),("interaction_personal","common_rank_personal")]
    comparisons = [{"comparison":new+"-"+old,**paired_uncertainty(test,nll[new]-nll[old])} for new,old in pairs]
    for row,p in zip(comparisons,holm([row["normal_approximation_p"] for row in comparisons])):
        row["holm_p_three_primary"] = p
    calibration_counts = anchors["test"].groupby("rater_id").size()
    prediction_rows = test_all.copy()
    prediction_rows["anchor_count"] = prediction_rows.rater_id.map(calibration_counts).fillna(0).astype(int)
    prediction_rows["primary_women_men"] = primary_mask
    for name,p in all_probabilities.items():
        prediction_rows[name+"_nll"] = losses(test_all,p)
        for label in range(3):
            prediction_rows[f"{name}_p{label}"] = p[:,label]
    prediction_rows.to_parquet(args.output/"heldout_predictions.parquet",index=False)
    support = []
    for keys,frame in prediction_rows.groupby(["reconstructed_gender","anchor_count"],dropna=False):
        support.append({"gender":keys[0],"anchor_count":int(keys[1]),"ratings":len(frame),"raters":frame.rater_id.nunique(),
                        **{name+"_nll":float(np.average(frame[name+"_nll"],weights=rater_weights(frame))) for name in predicted}})
    pd.DataFrame(support).to_csv(args.output/"calibration_support.csv",index=False)
    exactly_two = test.rater_id.map(calibration_counts).eq(2).to_numpy()
    exact_frame = test[exactly_two].reset_index(drop=True)
    exactly_two_comparisons = [{"comparison":new+"-"+old,**paired_uncertainty(exact_frame,(nll[new]-nll[old])[exactly_two])}
                              for new,old in pairs]
    secondary = []
    for rater_split,text_split in [("test","train"),("train","test")]:
        cell = responses[responses.rater_split.eq(rater_split)&responses.text_split.eq(text_split)&
                         responses.reconstructed_gender.isin(["women","men"])].reset_index(drop=True)
        calibrators = responses[responses.rater_split.eq(rater_split)&responses.calibration].reset_index(drop=True)
        updated,record = personalize(matrix(x,calibrators),calibrators,shared,None,precision)
        fits.append({"stage":"secondary_crossed_personal","rater_split":rater_split,**record})
        fixed = probabilities(matrix(x,cell),cell,shared)
        adapted = probabilities(matrix(x,cell),cell,shared,personal=updated)
        secondary.append({"rater_split":rater_split,"text_split":text_split,"rows":len(cell),
                          "raters":cell.rater_id.nunique(),"texts":cell.item_id.nunique(),
                          "pooled_nll":mean_loss(cell,fixed),"personal_nll":mean_loss(cell,adapted),
                          "training_raters_have_population_fit_label_exposure":rater_split=="train"})
    refit_values = {new+"-"+old:[] for new,old in pairs[1:]}
    rng = np.random.default_rng(SEED)
    people = np.unique(train.rater_id)
    for draw in range(args.training_refits):
        multiplicities = dict(zip(people,rng.multinomial(len(people),np.full(len(people),1/len(people)))))
        s = fit_shared(xt,train,selected_ridge,multiplicities,shared["parameters"])
        g = fit_rank_head(xt,train,s,multiplicity=multiplicities,initial=common_rank["parameters"])
        interaction_fit = fit_rank_head(xt,train,s,interaction_ridge,multiplicities,interaction["parameters"])
        a,_ = personalize(ae,anchors["test"],s,None,precision)
        b,_ = personalize(ae,anchors["test"],s,g,precision)
        c,_ = personalize(ae,anchors["test"],s,interaction_fit,precision)
        refit_values["personal-pooled"].append(mean_loss(test,probabilities(xe,test,s,personal=a))-mean_loss(test,probabilities(xe,test,s)))
        refit_values["interaction_personal-common_rank_personal"].append(mean_loss(test,probabilities(xe,test,s,interaction_fit,c))-
                                                                 mean_loss(test,probabilities(xe,test,s,g,b)))
        fits.extend([{ "stage":"training_refit","draw":draw,**f["record"]} for f in (s,g,interaction_fit)])
        if (draw+1)%10==0:
            print(json.dumps({"training_refits_completed":draw+1,"requested":args.training_refits}),flush=True)
    refit_summary = {name:{"draws":len(v),"training_only_95_interval":np.quantile(v,[.025,.975]).tolist() if v else None,
                           "standard_error":float(np.std(v,ddof=1)) if len(v)>1 else None} for name,v in refit_values.items()}
    np.savez_compressed(args.output/"training_refit_draws.npz",**{name:np.asarray(v) for name,v in refit_values.items()})
    np.savez_compressed(args.output/"population_parameters.npz",shared=shared["parameters"],group=group["parameters"],
                        common_rank=common_rank["parameters"],interaction=interaction["parameters"],TFIDF=lexical_model["parameters"])
    (args.output/"fits.json").write_text(json.dumps(fits,indent=2)+"\n")
    (args.output/"summary.json").write_text(json.dumps({"model":args.model_name,"embeddings":str(args.embeddings),
        "metrics":metrics,"comparisons":comparisons,"training_refit":refit_summary,
        "exactly_two_anchor_rows":len(exact_frame),"exactly_two_anchor_comparisons":exactly_two_comparisons,
        "secondary_crossed_transfer":secondary,
        "training_refit_scope":"training-rater cluster reweighting; population heads refitted; original PCA/vocabulary, hyperparameters and test anchors fixed; excludes TFIDF contrast",
        "test_rows":len(test),"test_people":test.rater_id.nunique(),"test_texts":test.item_id.nunique(),
        "test_class_counts":{str(k):int(v) for k,v in test.hatespeech.value_counts().sort_index().items()},
        "ordinal_assumption":"no/unclear/yes treated as ordered response categories, not a measured continuous hate amount",
        "penalty_convention":"rater-equal weights rescaled to multiplicity-expanded training row count; half ridge times squared coefficients in weighted sum loss",
        "complete":True},indent=2)+"\n")


if __name__=="__main__":
    main()
