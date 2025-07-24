#include "rela/transition.h"

using namespace rela;

MultiStepTransition::MultiStepTransition(const MultiStepTransition& tau, int bsz) {
  batchFirst_ = true;

  obs = tensor_dict::allocateBatchStorage(tau.obs, bsz);
  bc_obs = tensor_dict::allocateBatchStorage(tau.bc_obs, bsz);
  next_bc_obs = tensor_dict::allocateBatchStorage(tau.next_bc_obs, bsz);
  action = tensor_dict::allocateBatchStorage(tau.action, bsz);
  h0 = tensor_dict::allocateBatchStorage(tau.h0, bsz);
  reward = torch::zeros(tensor_dict::getBatchedSize(tau.reward, bsz));
  bootstrap = torch::zeros(tensor_dict::getBatchedSize(tau.bootstrap, bsz));
  seqLen = torch::zeros(bsz);
}

void MultiStepTransition::paste_(const MultiStepTransition& tau, int idx) {
  assert(batchFirst_);

  for (auto& kv : tau.obs) {
    obs[kv.first][idx] = kv.second;
  }
  for (auto& kv : tau.bc_obs) {
    bc_obs[kv.first][idx] = kv.second;
  }
  for (auto& kv : tau.next_bc_obs) {
    next_bc_obs[kv.first][idx] = kv.second;
  }
  for (auto& kv : tau.action) {
    action[kv.first][idx] = kv.second;
  }
  for (auto& kv : tau.h0) {
    h0[kv.first][idx] = kv.second;
  }
  reward[idx] = tau.reward;
  bootstrap[idx] = tau.bootstrap;
  seqLen[idx] = tau.seqLen;
}

MultiStepTransition MultiStepTransition::index(int i) const {
  // assert(batchFirst_);
  MultiStepTransition element;

  for (auto& name2tensor : obs) {
    element.obs.insert({name2tensor.first, name2tensor.second[i]});
  }
  for (auto& name2tensor : bc_obs) {
    element.bc_obs.insert({name2tensor.first, name2tensor.second[i]});
  }
  for (auto& name2tensor : next_bc_obs) {
    element.next_bc_obs.insert({name2tensor.first, name2tensor.second[i]});
  }
  for (auto& name2tensor : h0) {
    element.h0.insert({name2tensor.first, name2tensor.second[i]});
  }
  for (auto& name2tensor : action) {
    element.action.insert({name2tensor.first, name2tensor.second[i]});
  }

  element.reward = reward[i];
  element.bootstrap = bootstrap[i];
  element.seqLen = seqLen[i];
  return element;
}

void MultiStepTransition::copyTo(int from, MultiStepTransition& dst, int to) const {
  assert(batchFirst_);
  assert(dst.batchFirst_);

  for (auto& kv : obs) {
    dst.obs[kv.first][to] = kv.second[from];
  }
  for (auto& kv : bc_obs) {
    dst.bc_obs[kv.first][to] = kv.second[from];
  }
  for (auto& kv : next_bc_obs) {
    dst.next_bc_obs[kv.first][to] = kv.second[from];
  }
  for (auto& kv : h0) {
    dst.h0[kv.first][to] = kv.second[from];
  }
  for (auto& kv : action) {
    dst.action[kv.first][to] = kv.second[from];
  }

  dst.reward[to] = reward[from];
  dst.bootstrap[to] = bootstrap[from];
  dst.seqLen[to] = seqLen[from];
}

void MultiStepTransition::to_(const std::string& device) {
  if (device == "cpu") {
    return;
  }

  auto d = torch::Device(device);
  auto toDevice = [&](const torch::Tensor& t) { return t.to(d); };
  obs = tensor_dict::apply(obs, toDevice);
  bc_obs = tensor_dict::apply(bc_obs, toDevice);
  next_bc_obs = tensor_dict::apply(next_bc_obs, toDevice);
  h0 = tensor_dict::apply(h0, toDevice);
  action = tensor_dict::apply(action, toDevice);
  reward = reward.to(d);
  bootstrap = bootstrap.to(d);
  seqLen = seqLen.to(d);
}

void MultiStepTransition::seqFirst_() {
  assert(batchFirst_);
  batchFirst_ = false;

  for (auto& kv : obs) {
    obs[kv.first] = kv.second.transpose(0, 1).contiguous();
  }
  for (auto& kv : bc_obs) {
    bc_obs[kv.first] = kv.second.transpose(0, 1).contiguous();
  }
  for (auto& kv : next_bc_obs) {
    next_bc_obs[kv.first] = kv.second.transpose(0, 1).contiguous();
  }
  for (auto& kv : h0) {
    h0[kv.first] = kv.second.transpose(0, 1).contiguous();
  }
  for (auto& kv : action) {
    action[kv.first] = kv.second.transpose(0, 1).contiguous();
  }
  reward = reward.transpose(0, 1).contiguous();
  bootstrap = bootstrap.transpose(0, 1).contiguous();
  // no need to transpose seqLen
}

MultiStepTransition rela::makeBatch(
    const std::vector<MultiStepTransition>& transitions, const std::string& device) {
  std::vector<TensorDict> obsVec;
  std::vector<TensorDict> h0Vec;
  std::vector<TensorDict> actionVec;
  std::vector<TensorDict> bc_obsVec;
  std::vector<TensorDict> next_bc_obsVec;
  std::vector<torch::Tensor> rewardVec;
  std::vector<torch::Tensor> bootstrapVec;
  std::vector<torch::Tensor> seqLenVec;

  for (size_t i = 0; i < transitions.size(); i++) {
    obsVec.push_back(transitions[i].obs);
    h0Vec.push_back(transitions[i].h0);
    actionVec.push_back(transitions[i].action);
    bc_obsVec.push_back(transitions[i].bc_obs);
    next_bc_obsVec.push_back(transitions[i].next_bc_obs);
    rewardVec.push_back(transitions[i].reward);
    bootstrapVec.push_back(transitions[i].bootstrap);
    seqLenVec.push_back(transitions[i].seqLen);
  }

  MultiStepTransition batch;
  batch.obs = tensor_dict::stack(obsVec, 1);
  batch.h0 = tensor_dict::stack(h0Vec, 1);  // 1 is batch for rnn hid
  batch.action = tensor_dict::stack(actionVec, 1);
  batch.bc_obs = tensor_dict::stack(bc_obsVec, 1);
  batch.next_bc_obs = tensor_dict::stack(next_bc_obsVec, 1);
  batch.reward = torch::stack(rewardVec, 1);
  batch.bootstrap = torch::stack(bootstrapVec, 1);
  batch.seqLen = torch::stack(seqLenVec, 0);

  if (device != "cpu") {
    auto d = torch::Device(device);
    auto toDevice = [&](const torch::Tensor& t) { return t.to(d); };
    batch.obs = tensor_dict::apply(batch.obs, toDevice);
    batch.h0 = tensor_dict::apply(batch.h0, toDevice);
    batch.action = tensor_dict::apply(batch.action, toDevice);
    batch.bc_obs = tensor_dict::apply(batch.bc_obs, toDevice);
    batch.next_bc_obs = tensor_dict::apply(batch.next_bc_obs, toDevice);
    batch.reward = batch.reward.to(d);
    batch.bootstrap = batch.bootstrap.to(d);
    batch.seqLen = batch.seqLen.to(d);
  }

  return batch;
}

SingleStepTransition rela::makeBatch(
    const std::vector<SingleStepTransition>& transitions, const std::string& device) {
  std::vector<TensorDict> obsVec;
  std::vector<TensorDict> nextObsVec;
  std::vector<TensorDict> bc_obsVec;
  std::vector<TensorDict> next_bc_obsVec;
  std::vector<TensorDict> actionVec;
  std::vector<torch::Tensor> rewardVec;
  std::vector<torch::Tensor> bootstrapVec;

  for (size_t i = 0; i < transitions.size(); i++) {
    obsVec.push_back(transitions[i].obs);
    nextObsVec.push_back(transitions[i].nextObs);
    bc_obsVec.push_back(transitions[i].bc_obs);
    next_bc_obsVec.push_back(transitions[i].next_bc_obs);
    actionVec.push_back(transitions[i].action);
    rewardVec.push_back(transitions[i].reward);
    bootstrapVec.push_back(transitions[i].bootstrap);
  }

  SingleStepTransition batch;
  batch.obs = tensor_dict::stack(obsVec, 0);
  batch.nextObs = tensor_dict::stack(nextObsVec, 0);
  batch.bc_obs = tensor_dict::stack(bc_obsVec, 0);
  batch.next_bc_obs = tensor_dict::stack(next_bc_obsVec, 0);
  batch.action = tensor_dict::stack(actionVec, 0);
  batch.reward = torch::stack(rewardVec, 0);
  batch.bootstrap = torch::stack(bootstrapVec, 0);

  if (device != "cpu") {
    auto d = torch::Device(device);
    auto toDevice = [&](const torch::Tensor& t) { return t.to(d); };
    batch.obs = tensor_dict::apply(batch.obs, toDevice);
    batch.nextObs = tensor_dict::apply(batch.nextObs, toDevice);
    batch.bc_obs = tensor_dict::apply(batch.bc_obs, toDevice);
    batch.next_bc_obs = tensor_dict::apply(batch.next_bc_obs, toDevice);
    batch.action = tensor_dict::apply(batch.action, toDevice);
    batch.reward = batch.reward.to(d);
    batch.bootstrap = batch.bootstrap.to(d);
  }
  return batch;
}
