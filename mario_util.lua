require "torch"

local LoopQueue = {}

function LoopQueue:new(max_n)
  max_n = math.max(1, max_n)
  local o = {
    _max_n = max_n,
    _full = false,
    _next = 1
  }
  setmetatable(o, self)
  self.__index = self
  return o
end

function LoopQueue:append(x)
  if not self._full and self._next == self._max_n then
    self._full = true
  end
  self[self._next] = x
  self._next = self._next % self._max_n + 1
end

function LoopQueue:isFull()
  return self._full
end

function LoopQueue:maxSize()
  return self._max_n
end

function LoopQueue:size()
  return self._full and self._max_n or self._next - 1
end

function LoopQueue:at(i)
  if self._full and i <= self._max_n then
    return self[(self._next + i - 2) % self._max_n + 1]
  elseif not self._full and i <= self._next - 1 then
    return self[i]
  end
end

function LoopQueue:array()
  local result = {}
  for i, v in self:xarray() do
    result[#result + 1] = v
  end
  return result
end

function LoopQueue:xarray()
  local i = 0
  return function ()
    i = i + 1
    v = self:at(i)
    if v then
      return i, v
    end
  end
end

local function bool2IntArray(a)
  local result = {}
  for k, v in pairs(a) do
    result[k] = v and 1 or 0
  end
  return result
end

local function hasBit(x, p)
  return x % (p + p) >= p
end

local function reverseTable(t)
  local result = {}
  for k, v in pairs(t) do
    result[v] = k
  end
  return result
end

local function decodeJoypadInput(input_code)
  -- input_code is an 8-bit integer in [0, 255]
  -- |start|select|up|down|left|right|A|B|
  return {
    start = hasBit(input_code, 0x80),
    select = hasBit(input_code, 0x40),
    up = hasBit(input_code, 0x20),
    down = hasBit(input_code, 0x10),
    left = hasBit(input_code, 0x08),
    right = hasBit(input_code, 0x04),
    A = hasBit(input_code, 0x02),
    B = hasBit(input_code, 0x01),
  }
end

local function joypadInputToString(input)
  input = bool2IntArray(input)
  ss = {""}
  for i, key in ipairs{
    "start", "select", "up", "down", "left", "right", "A", "B"} do
    ss[#ss + 1] = string.format("%s=%d", key, input[key])
  end
  ss[#ss + 1] = ""
  return table.concat(ss, "|")
end

local function actionToString(action)
  return joypadInputToString(decodeJoypadInput(action))
end

function log(f, msg)
  if f then
    print(msg)
    f:write(msg.."\n")
    f:flush()
  end
end

function randSample(m, n)
  -- random sample m numbers from 1..n without replacement
  m = math.min(m, n)
  local result = {}
  local result_set = {}
  for i = 1,m do
    local i = torch.random(1, n)
    while result_set[i] do
      i = torch.random(1, n)
    end
    result[#result + 1] = i
    result_set[i] = true
  end
  return result
end

function permute(tab)
  local n = #tab
  for i = 1, n do
    local j = torch.random(i, n)
    tab[i], tab[j] = tab[j], tab[i]
  end
  return tab
end

mario_util = {
  LoopQueue = LoopQueue,
  bool2IntArray = bool2IntArray,
  decodeJoypadInput = decodeJoypadInput,
  joypadInputToString = joypadInputToString,
  actionToString = actionToString,
  reverseTable = reverseTable,
  log = log,
  randSample = randSample,
  permute = permute,
}
return mario_util
