require "torch"
require "math"

require "mario_util"
require "mario_game"

local _sandbox = mario_game.sandbox

function _actionToString(a)
  return mario_util.joypadInputToString(
    mario_util.decodeJoypadInput(a))
end

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
    num_skip_frames = 12,
    min_num_visits_to_expand_node = 1,
    max_num_runs = 100,
    max_depth = 80,    
    uct_const = 100.0,
    enable_debug = true,
    
    save_to = nil,
    log_file = nil,
    result_actions = {},
    
    _nodes = {},
    _num_saves = 0,
    _result_action_cursor = 1,
    
    _depth = nil,
    _root_node = nil,
    _root_stats = nil,
    _run_trace = nil,
    _save = nil,
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
  self:_debugMessage(_actionToString(action))
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
    mean_x2 = 0.0,
    max_x = nil,
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
  _sandbox:startGame(self._save)
  
  local num_result_actions = #self.result_actions
  local played = false
  while not _sandbox:isGameOver() and
        self._result_action_cursor <= num_result_actions do
    self:_takeAction(self.result_actions[self._result_action_cursor])
    self._result_action_cursor = self._result_action_cursor + 1
    played = true
  end

  if played or not self._save then
    self._save = _sandbox:saveGame()
  end

  self._depth = 0
  self._root_node = self:_getNode(self:_getState())
  self._root_stats = _sandbox:getMarioStats()
  self._run_trace = {}
  self:_appendRunTrace(nil, self._root_node)
end

function UctModel:_treePolicy(node)
  self:_debugMessage("treePolicy")
  while not _sandbox:isGameOver() do
    if node.num_arcs < #self.all_actions then
      if node == self._root_node or
         node.num_visits >= self.min_num_visits_to_expand_node then
        return self:_expandNode(node)
      else
        return node
      end
    end
    node = self:_bestChild(node)
  end
  self:_debugMessage("game over")
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
  local child_node = self:_getNode(self:_getState())
  local arc = self:_getArc(node, action)
  arc.child_node = child_node
  self:_appendRunTrace(arc, child_node)
  return child_node
end

function UctModel:_ucb(node, arc)
  assert(node.num_visits > 0 and arc.num_visits > 0, "ucb error: zero visits")
  return
    arc.mean_x +
    self.uct_const * math.sqrt(math.log(node.num_visits) / arc.num_visits)
end

function UctModel:_randActionAndArcs(node)
  local action_and_arcs = {}
  for a, arc in pairs(node.arcs) do
    table.insert(action_and_arcs, {a, arc})
  end
  return mario_util.permute(action_and_arcs)
end

function UctModel:_bestChild(node)
  self:_debugMessage("bestChild")
  local max_ucb = nil
  local key_action = nil
  local key_arc = nil
  local action_and_arcs = self:_randActionAndArcs(node)
  for i, aa in ipairs(action_and_arcs) do
    local a, arc = aa[1], aa[2]
    local ucb = self:_ucb(node, arc)
    if not max_ucb or max_ucb < ucb then
      max_ucb, key_action, key_arc = ucb, a, arc
    end
  end
  self:_takeAction(key_action)
  local child_node = key_arc.child_node
  self:_appendRunTrace(key_arc, child_node)
  return child_node
end

function UctModel:_defaultPolicy(node)
  self:_debugMessage("defaultPolicy")
  while not _sandbox:isGameOver() do
    if self._depth >= self.max_depth then
      break
    end
    self._depth = self._depth + 1
    local action = self.all_actions[torch.random(1, #self.all_actions)]
    self:_takeAction(action)
  end
  return _sandbox:getMarioStats()
end

function UctModel:_estimateTerminalScore(terminal_stats)
  return
    (terminal_stats.score - self._root_stats.score) +
    (terminal_stats.is_game_over and 0.0 or 10.0)
end

function UctModel:_backup(terminal_stats)
  self:_debugMessage("backup")
  for i, t in ipairs(self._run_trace) do
    local arc, child_node = t[1], t[2]
    if arc then
      arc.num_visits = arc.num_visits + 1
      local terminal_score = self:_estimateTerminalScore(terminal_stats)
      arc.mean_x =
        arc.mean_x +
        (terminal_score - arc.mean_x) * 1.0 / arc.num_visits
      arc.mean_x2 =
        arc.mean_x2 +
        (terminal_score * terminal_score - arc.mean_x) * 1.0 / arc.num_visits
      if not arc.max_x or arc.max_x < terminal_score then
        arc.max_x = terminal_score
      end
    end
    child_node.num_visits = child_node.num_visits + 1
  end
end

function UctModel:_bestAction(node)
  local max_num_visits = nil
  local key_action = nil
  local action_and_arcs = self:_randActionAndArcs(node)
  for i, aa in ipairs(action_and_arcs) do
    local a, arc = aa[1], aa[2]
    if not max_num_visits or max_num_visits < arc.num_visits then
      max_num_visits, key_action = arc.num_visits, a
    end
  end
  return key_action
end

function UctModel:_search()
  self:_debugMessage("search")
  local num_runs = 0
  while num_runs < self.max_num_runs do
    num_runs = num_runs + 1
    self:_debugMessage(string.format("run #%d", num_runs))
    self:_startSearch()
    local node = self:_treePolicy(self._root_node)
    local terminal_stats = self:_defaultPolicy(node)
    self:_backup(terminal_stats)
    self:_debugNodes()
  end
  local best_action = self:_bestAction(self._root_node)
  if not best_action then
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
  self:_log("================= debug ================")
  self:_log("result actions: ")
  for i, a in ipairs(self.result_actions) do
    self:_log(_actionToString(a))
  end
  local node_count = 0
  for s, node in pairs(self._nodes) do
    self:_log(string.format("node #%d", node_count + 1))
    self:_log(string.format("  num_visits = %d", node.num_visits))
    for a, arc in pairs(node.arcs) do
      self:_log(string.format("  arc a = %s", _actionToString(a)))
      self:_log(string.format("    num_visits = %d", arc.num_visits))
      self:_log(string.format("    mean_x     = %.2f", arc.mean_x))
      self:_log(string.format("    mean_x2    = %.2f", arc.mean_x2))
      self:_log(string.format("    max_x      = %.2f", arc.max_x))
    end
    node_count = node_count + 1
  end
  self:_log(string.format("#nodes = %d", node_count))
  self:_log("========================================")
end

function UctModel:main()
  self:_log("UCT Main")
  while self:_search() do
  end
end

mario_uct_model = {
  UctModel = UctModel
}
return mario_uct_model
