import torch
import numpy as np


def simi_direction_suppression(relation_prediction, relation_conf_thres=0.01, dire_thres=0.45,relation_count_limit = 500):

    nc = relation_prediction.shape[2] - 3  # number of relation classes
    xc = relation_prediction[..., 0] > relation_conf_thres  # candidates
    #print(relation_prediction.size())
    #relation_prediction = relation_prediction.view()

    relation_output = [torch.zeros((0, 6), device=relation_prediction.device)] * relation_prediction.shape[0]

    for xi, x in enumerate(relation_prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        direction = x[:,1:3].T[[1,0]].T

        #print(direction)
        #bias = x[:,3:5].T[[1,0]].T
        #print(direction)
        local = x[:,3:5]
        local = local#+bias
        #print(local)
        #print(local+direction)
        relation_cls = x[:,5:]#* x[:,0:1]
        conf, j = relation_cls.max(1, keepdim=True)

        x_r = torch.tensor([xi],device=relation_prediction.device).repeat(len(j),1)
        #x = torch.cat((x_r,local, local+direction, j.float(),conf, relation_cls), 1)[conf.view(-1) > relation_conf_thres]
        x = torch.cat((x_r, local, local + direction, j.float(), x[:,0:1], relation_cls), 1)

        x = x[x[:, 6].argsort(descending=True)]
        # suppression_direction
        # if len(x)>0:
        #     x = suppression_direction(x)
        #     #x = x[:, :6]
        # else:
        #     x = x[:,:6]

        relation_output[xi]=x

    relation_output = torch.cat(relation_output,0)

    if len(relation_output)>relation_count_limit:
        relation_output = relation_output[:relation_count_limit]

    return relation_output

def relation_search(targets,relations,names,relation_names,distance_threshold=48,object_scores=None):
    if len(relations)==0:
        return []

    n = relations[...,0].max()

    current_targets = targets[targets[...,0]==0]
    target_number = torch.from_numpy(np.arange(len(current_targets))).unsqueeze(1).to(device=current_targets.device)
    current_targets = torch.cat((current_targets,target_number),dim = 1)
    current_relations = relations[relations[...,0]==0]
    # current_targets ,image,cls,x,y,w,h,target_number
    # current_relations, image,x1,y1, x2,y2, relation_cls,relation_confidence,relation_scores(51)

    #triplet : subject,object,relation_cls,relation_confidence, relation_scores(51)

    triplet = batch_close_selection(current_targets, current_relations, distance_threshold=distance_threshold, object_scores = object_scores)


    return triplet



def batch_close_selection(current_targets,current_relations,distance_threshold = 10,object_scores=None):

    n_r = len(current_relations)
    n_t = len(current_targets)

    ai = torch.arange(n_r, device=current_targets.device).float().view(n_r, 1).repeat(1, n_t)  # same as .repeat_interleave(nt)
    current_targets = current_targets.repeat(n_r, 1, 1)
    current_targets = torch.cat((current_targets, ai[:, :, None]), 2)# append anchor indices

    sub_delta_x = torch.abs(current_targets[..., 2] - current_relations[..., 1].unsqueeze(1))
    sub_delta_y = torch.abs(current_targets[..., 3] - current_relations[..., 2].unsqueeze(1))
    sub_delta = torch.norm(torch.stack((sub_delta_x, sub_delta_y), dim=1), p=2, dim=1)
    sub_arg = sub_delta.argmin(dim=1)
    b_i = torch.arange(n_r, device=current_targets.device)
    sub_candidate_targets = current_targets[b_i,sub_arg,None]

    obj_delta_x = torch.abs(current_targets[..., 2] - current_relations[..., 3].unsqueeze(1))
    obj_delta_y = torch.abs(current_targets[..., 3] - current_relations[..., 4].unsqueeze(1))
    obj_delta = torch.norm(torch.stack((obj_delta_x, obj_delta_y), dim=1), p=2, dim=1)
    obj_arg = obj_delta.argmin(dim =1)
    obj_candidate_targets = current_targets[b_i, obj_arg, None]

    _candidate = (sub_delta[b_i,sub_arg,None]<distance_threshold) & (obj_delta[b_i,obj_arg,None]<distance_threshold)

    sub_candidate_targets = sub_candidate_targets[_candidate]
    obj_candidate_targets = obj_candidate_targets[_candidate]
    candidate_relations = current_relations.unsqueeze(1)[_candidate]

    triplet = torch.stack((sub_candidate_targets[:,6],obj_candidate_targets[:,6],candidate_relations[:,5]),dim=1)

    triplet = torch.cat((triplet,candidate_relations[:,6:]),dim=1)

    if object_scores!=None:
        probability = object_scores[sub_candidate_targets[:, 6].long()]*object_scores[obj_candidate_targets[:, 6].long()]*candidate_relations[:,6]
        pro_soreted,pro_indecs = probability.sort(descending=True)
        triplet = triplet[pro_indecs]

    _triplet, _ = torch.unique(triplet[:,:2],dim=0,return_inverse=True)

    _ = _.cpu().numpy()
    vals, idx_start= np.unique(np.array(_), return_index=True)

    #__, a_ = torch.unique(_,dim=-1,  return_inverse=True)
    # print(vals)
    triplet = triplet[torch.tensor(idx_start)]

    return triplet


def objects_search_relation(targets,relations,names,relation_names,distance_threshold=32,object_scores=None):
    '''
        object_scores: if using object_scores, triplet sorted by object_confidence*relation_confidence; elif object_scores=None, triplet only sorted by relation_confidence.
    '''
    if len(relations)==0:
        return []

    n = relations[...,0].max()

    current_targets = targets[targets[...,0]==0]
    target_number = torch.from_numpy(np.arange(len(current_targets))).unsqueeze(1).to(device=current_targets.device)
    current_targets = torch.cat((current_targets,target_number),dim = 1)
    current_relations = relations[relations[...,0]==0]

    # current_targets ,image,cls,x,y,w,h,target_number
    # current_relations, image,x1,y1, x2,y2, relation_cls,relation_confidence,relation_scores(51)

    #triplet : subject,object,relation_num,relation_confidence, relation_scores(51)
    triplet = batch_objects_pair_selection(current_targets, current_relations,distance_threshold = distance_threshold,object_scores=object_scores)

    return triplet

def batch_objects_pair_selection(current_targets,current_relations,distance_threshold=32,object_scores=None):
    n_r = len(current_relations)
    n_t = len(current_targets)
    #print(n_r,n_t)
    ai = torch.arange(n_t, device=current_targets.device).float().view(n_t, 1)
    #print(ai.size())
    current_targets = torch.cat((current_targets,ai),dim=1)
    current_targets = current_targets.repeat(n_r, 1, 1)
    #print(current_targets.size(),current_relations.size())

    sub_delta_x = torch.abs(current_targets[..., 2] - current_relations[..., 1].unsqueeze(1)).transpose(0,1)
    sub_delta_y = torch.abs(current_targets[..., 3] - current_relations[..., 2].unsqueeze(1)).transpose(0,1)
    sub_delta = torch.sqrt(sub_delta_x**2+sub_delta_y**2)
    sub_delta = torch.cat((ai,sub_delta),dim=1) #[object_id, object2each_realtion_sub_distance]
    batch_sub_delta = sub_delta.repeat(len(sub_delta),1)

    obj_delta_x = torch.abs(current_targets[..., 2] - current_relations[..., 3].unsqueeze(1)).transpose(0,1)
    obj_delta_y = torch.abs(current_targets[..., 3] - current_relations[..., 4].unsqueeze(1)).transpose(0,1)
    obj_delta = torch.sqrt(obj_delta_x**2+obj_delta_y**2)
    obj_delta = torch.cat((ai,obj_delta),dim=1) #[object_id, object2each_realtion_obj_distance]
    batch_obj_delta = obj_delta.repeat(1,len(obj_delta)).view(-1,len(obj_delta[0]))

    results = torch.stack((batch_sub_delta[:,0],batch_obj_delta[:,0]),dim=1)
    results = torch.cat((results,batch_sub_delta[:,1:]+batch_obj_delta[:,1:]),dim=1)
    _relation,relation_select = results[:, 2:].min(dim=1)
    _relation = _relation<(distance_threshold)

    #relation_select = results[:,2:].argmin(dim=1)
    # for r in relation_select:
    #     print(r)
    # input()
    candidate_relations = current_relations[relation_select]

    #sub,obj
    triplet = torch.cat((results[:,:2],candidate_relations[:,5:]),dim=1)
    triplet = triplet[_relation]


    if object_scores!=None:
        probability = object_scores[triplet[:, 0].long()]*object_scores[triplet[:, 1].long()]*triplet[:,3]
        pro_soreted,pro_indecs = probability.sort(descending=True)
        triplet = triplet[pro_indecs]
    #triplet = torch.cat((results[:, :2], candidate_relations[:, 5:10]), dim=1)

    return triplet