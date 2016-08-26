require "torch"
require "math"

require "mario_util"
require "mario_game"

local _sandbox = mario_game.sandbox

local UctModel = {}

function UctModel:new()
  local o = {
    all_actions = {
      0x00, -- nil
      0x08, -- left
      0x04, -- right
      0x20, -- up
      0x10, -- down
      0x02, -- A
      0x01, -- B
      0x0A, -- left + A
      0x09, -- left + B
      0x06, -- right + A
      0x05, -- right + B
    },
    
    game_save = nil,
    result_actions = {},
    num_skip_frames = 12,
    
    min_num_visits_to_expand_node = 1,
    max_num_runs = 100,
    max_depth = 60,
    mario_score_ceil = 200,
    use_ucb1 = true,
    
    save_to = nil,
    log_file = nil,
    enable_debug = true,
    
    _nodes = {},
    _num_saves = 0,
    _result_action_cursor = 1,
    
    _depth = nil,
    _root_node = nil,
    _root_mario_stats = nil,
    _run_trace = nil,
  }
  
  setmetatable(o, self)
  self.__index = self
  return o
end

function UctModel:_getState()
  return _sandbox:getRam(true)
end

function UctModel:_takeAction(action)
  assert(action, "action must not be nil")
  _sandbox:advance(action, self.num_skip_frames)
  self:_debugMessage(string.format("depth = %d", self._depth or -1))
  self:_debugMessage(mario_util.actionToString(action))
end

function UctModel:_newNode(state)
  return {
    state = state,
    num_visits = 0,
    arcs = {},  -- indexed by action
    num_arcs = 0,
  }
end

function UctModel:_newArc()
  return {
    num_visits = 0,
    mean_x = 0.0,
    var_x = 0.0,
    max_x = 0.0,
    child_node = nil,
  }
end

function UctModel:_getNode(state)
  local node = self._nodes[state]
  if not node then
    node = self:_newNode(state)
    self._nodes[state] = node
  end
  return node
end

function UctModel:_getArc(node, action)
  local arc = node.arcs[action]
  if not arc then
    arc = self:_newArc()
    node.arcs[action] = arc
    node.num_arcs = node.num_arcs + 1
  end
  return arc
end

function UctModel:_appendRunTrace(arc, child_node)
  assert(child_node, "child node must not be nil")
  self:_debugMessage("appendRunTrace")
  table.insert(self._run_trace, {arc, child_node})
end

function UctModel:_appendResultAction(action)
  assert(action, "result action must not be nil")
  self:_debugMessage("appendResultAction")
  table.insert(self.result_actions, action)
end

function UctModel:_startSearch()
  self:_debugMessage("startSearch")
  _sandbox:startGame(self.game_save)
  
  local num_result_actions = #self.result_actions
  local played = false
  while not _sandbox:isGameOver() and
        self._result_action_cursor <= num_result_actions do
    self:_takeAction(self.result_actions[self._result_action_cursor])
    self._result_action_cursor = self._result_action_cursor + 1
    played = true
  end

  if played or not self.game_save then
    self.game_save = _sandbox:saveGame()
  end

  self._depth = 0
  self._root_node = self:_getNode(self:_getState())
  self._root_mario_stats = _sandbox:getMarioStats()
  self._run_trace = {}
  self:_appendRunTrace(nil, self._root_node)
end

function UctModel:_treePolicy(node)
  self:_debugMessage("treePolicy")
  while not _sandbox:isGameOver() and self._depth < self.max_depth do
    if node.num_arcs < #self.all_actions then
      if node == self._root_node or
         node.num_visits >= self.min_num_visits_to_expand_node then
        self._depth = self._depth + 1
        return self:_expandNode(node)
      else
        return node
      end
    end
    self._depth = self._depth + 1
    node = self:_bestChild(node)
  end
  return node
end

function UctModel:_getUntriedActions(node)
  if node.num_arcs == 0 then
    return self.all_actions
  end
  local untried_actions = {}
  for i, a in ipairs(self.all_actions) do
    if not node.arcs[a] then
      table.insert(untried_actions, a)
    end
  end
  return untried_actions
end

function UctModel:_expandNode(node)
  self:_debugMessage("expandNode")
  local untried_actions = self:_getUntriedActions(node)
  local action = untried_actions[torch.random(1, #untried_actions)]
  self:_takeAction(action)
  -- create new node
  local child_node = self:_getNode(self:_getState())
  -- create new arc
  local arc = self:_getArc(node, action)
  arc.child_node = child_node

  self:_appendRunTrace(arc, child_node)
  return child_node
end

local function _ucb(node, arc)
  assert(node.num_visits > 0 and arc.num_visits > 0, "ucb error: zero visits")
  return
    arc.mean_x + math.sqrt(math.log(node.num_visits) * 2.0 / arc.num_visits)
end

local function _ucb1(node, arc)
  assert(node.num_visits > 0 and arc.num_visits > 0, "ucb error: zero visits")
  local tmp = math.log(node.num_visits) / arc.num_visits
  local result = math.max(0.0, arc.var_x) + math.sqrt(2.0 * tmp)
  result = arc.mean_x + math.sqrt(tmp * math.min(0.25, result))
  return result
end

local function _randActionAndArcs(node)
  local action_and_arcs = {}
  for a, arc in pairs(node.arcs) do
    table.insert(action_and_arcs, {a, arc})
  end
  return mario_util.permute(action_and_arcs)
end

local function _bestArc(node, metric_fn)
  -- metric_fn(node, arc) returns a number
  local max_metric = nil
  local key_action = nil
  local key_arc = nil
  local action_and_arcs = _randActionAndArcs(node)
  for i, aa in ipairs(action_and_arcs) do
    local a, arc = aa[1], aa[2]
    local metric = metric_fn(node, arc)
    if metric and (not max_metric or max_metric < metric) then
      max_metric, key_action, key_arc = metric, a, arc
    end
  end
  return max_metric, key_action, key_arc
end

function UctModel:_bestChild(node)
  self:_debugMessage("bestChild")
  local max_ucb, action, arc = _bestArc(node, self.use_ucb1 and _ucb1 or _ucb)
  self:_takeAction(action)
  local child_node = arc.child_node
  self:_appendRunTrace(arc, child_node)
  return child_node
end

function UctModel:_bestAction(node)
  self:_debugMessage("bestAction")
  local max_metric, action, arc = _bestArc(
    node, function(p_node, p_arc) return p_arc.max_x end)
  return action
end

function UctModel:_defaultPolicy(node)
  self:_debugMessage("defaultPolicy")
  while not _sandbox:isGameOver() and self._depth < self.max_depth do
    self._depth = self._depth + 1
    local action = self.all_actions[torch.random(1, #self.all_actions)]
    self:_takeAction(action)
  end
  return _sandbox:getMarioStats()
end

function UctModel:_estimateStateScore(mario_stats)
  local a, b = self._root_mario_stats, mario_stats
  
  -- score change
  local s = b.score - a.score
  
  -- still alive bonus
  s = s + (b.is_game_over and 0.0 or 10.0)

  -- forward displacement bonus
  local dx = 0
  if a.world ~= b.world or a.level ~= b.level then
    dx = 256 + b.x - a.x
  elseif a.block == 0 and b.block == 0 then  -- in bonus
    dx = 0
  elseif a.block == 0 then  -- leave bonus
    dx = b.x
  elseif b.block == 0 then  -- enter bonus
    dx = 256 - a.x
  elseif b.block > a.block then
    dx = 256 + b.x - a.x
  elseif b.block < a.block then
    dx = -256 + b.x - a.x
  else
    dx = b.x - a.x
  end
  s = s + 0.1 * dx
  
  -- bound to [0, mario_score_ceil]
  s = math.min(math.max(s, 0.0), self.mario_score_ceil)

  -- scale to [0, 1]
  return s * 1.0 / self.mario_score_ceil
end

function UctModel:_backup(mario_stats)
  self:_debugMessage("backup")
  local x = self:_estimateStateScore(mario_stats)
  for i, t in ipairs(self._run_trace) do
    local arc, child_node = t[1], t[2]
    if arc then
      arc.num_visits = arc.num_visits + 1
      local delta = x - arc.mean_x
      arc.mean_x = arc.mean_x + delta * 1.0 / arc.num_visits
      arc.var_x =
        arc.var_x  + (delta * (x - arc.mean_x) - arc.var_x) / arc.num_visits
      arc.max_x = math.max(arc.max_x, x)
    end
    child_node.num_visits = child_node.num_visits + 1
  end
end

function UctModel:_search()
  self:_debugMessage("////////// search //////////")
  local num_runs = 0
  while num_runs < self.max_num_runs do
    num_runs = num_runs + 1
    self:_debugMessage(string.format("run #%d", num_runs))
    self:_startSearch()
    local node = self:_treePolicy(self._root_node)
    local mario_stats = self:_defaultPolicy(node)
    self:_backup(mario_stats)
    self:_debugNodes()
  end
  local best_action = self:_bestAction(self._root_node)
  if not best_action then
    self:_debugMessage("cannot select a best action")
    return false
  end
  self:_appendResultAction(best_action)
  self._nodes = {}
  self:_saveModel()
  return true
end

function UctModel:_saveModel()
  if not self.save_to then
    return
  end
  self._num_saves = self._num_saves + 1
  local id = (self._num_saves - 1) % 5 + 1
  local model_save_to = self.save_to..".model."..id

  self:_log("Saving model to "..model_save_to)
  torch.save(model_save_to, {
    num_skip_frames = self.num_skip_frames,
    result_actions = self.result_actions,
  })
end

function UctModel:_log(msg)
  mario_util.log(self.log_file, msg)
end

function UctModel:_debugMessage(msg)
  if not self.enable_debug then
    return
  end
  print(msg)
end

function UctModel:_debugNodes()
  if not self.enable_debug then
    return
  end
  self:_log("---------- debug ----------")
  self:_log("result actions: ")
  for i, a in ipairs(self.result_actions) do
    self:_log(mario_util.actionToString(a))
  end

  local node_count = 0
  node_ids = {}
  for s, node in pairs(self._nodes) do
    node_count = node_count + 1
    node_ids[node] = node_count
  end
    
  for s, node in pairs(self._nodes) do
    print(string.format("node = #%d", node_ids[node]))
    print(string.format("  num_visits = %d", node.num_visits))
    for a, arc in pairs(node.arcs) do
      print(string.format("  arc a = %s", mario_util.actionToString(a)))
      print(string.format("    num_visits = %d", arc.num_visits))
      print(string.format("    mean_x     = %.2f", arc.mean_x))
      print(string.format("    var_x      = %.2f", arc.var_x))
      print(string.format("    max_x      = %.2f", arc.max_x))
      print(string.format("    child_node = #%d", node_ids[arc.child_node]))
    end
  end
  self:_log(string.format("# of nodes = %d", node_count))
  self:_log("---------------------------")
end

function UctModel:main()
  self:_log("********** BEGIN **********")
  while self:_search() do
  end
  self:_log("********** END ************")
end

mario_uct_model = {
  UctModel = UctModel
}
return mario_uct_model
