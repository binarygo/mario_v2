require "torch"
require "math"
md5 = require "md5"

require "mario_util"

local SCREEN_WIDTH = 256
local SCREEN_HEIGHT = 240

local function _setJoypad(player, input_code)
  local input = mario_util.decodeJoypadInput(input_code)
  joypad.set(player, input)
  return input
end

local function _getLiveScreenFlag()
  return memory.readbyte(0x0757)
end

local function _skipLiveScreen()
  while _getLiveScreenFlag() == 0 do
    emu.frameadvance()
  end
  while _getLiveScreenFlag() == 1 do
    emu.frameadvance()
  end
end

local sandbox = {}

function sandbox:saveGame()
  local save = savestate.object()
  savestate.save(save)
  return save
end

function sandbox:loadGame(save)
  savestate.load(save)
end

function sandbox:_init()
  self._mario_lives = nil
  self._is_game_over = false
end

function sandbox:_update()
  if self._is_game_over then
    return
  end
  local prev_mario_lives = self._mario_lives
  self._mario_lives = self:getMarioLives()
  self._is_game_over =
    (memory.readbyte(0x000E) == 6) or
    (prev_mario_lives and self._mario_lives < prev_mario_lives)
end

function sandbox:startGame(save)
  self:_init()
  emu.speedmode("normal")
  if save then
    self:loadGame(save)
  else
    for i = 1, 100 do
      emu.frameadvance()
    end
    _setJoypad(1, 0x80)
    _skipLiveScreen()
  end
  self:_update()
end

function sandbox:advance(action, num_skip_frames)
  for i = 1, num_skip_frames do
    if self:isGameOver() then
      return false
    end
    if action then
      _setJoypad(1, action)
    end
    emu.frameadvance()
    self:_update()
  end
  return true
end

function sandbox:setTime(t1, t2, t3)
  -- set time to 999
  memory.writebyte(0x07F8, t1)
  memory.writebyte(0x07F9, t2)
  memory.writebyte(0x07FA, t3)  
end

function sandbox:getWorld()
  return memory.readbyte(0x075F)
end

function sandbox:getLevel()
  return memory.readbyte(0x0760)
end

function sandbox:getMarioScore()
  local score = 0
  local score_addr = 0x07DD
  for i = 1, 6 do
    score = score * 10 + memory.readbyte(score_addr)
    score_addr = score_addr + 1
  end
  return score
end

function sandbox:getMarioLives()
  return memory.readbyte(0x075A)
end

function sandbox:getMarioSpeed()
  local v = memory.readbyte(0x0057)
  if v <= 40 then
    return v
  elseif v >= 216 then
    return v - 256
  end
  assert(false, "Invalid speed")
end

function sandbox:isGameOver()
  return self._is_game_over
end

function sandbox:getMarioStats()
  return {
    score = self:getMarioScore(),
    is_game_over = self:isGameOver(),
  }
end

function sandbox:getScreenshot()
  local raw_screen = torch.ByteStorage():string(gui.gdscreenshot())
  local w = SCREEN_WIDTH
  local h = SCREEN_HEIGHT
  return torch.reshape(
    torch.ByteTensor(
      raw_screen, 12,
      torch.LongStorage{h * w, 4})[{{},{2,4}}]:t(), 3, h, w):float() / 255.0
end

function sandbox:getRam(md5_hash)
  local s = memory.readbyterange(0x0000, 0x800)
  return md5_hash and md5.sum(s) or s
end

function sandbox:message(msg)
  emu.message(msg)
end

mario_game = {
  SCREEN_WIDTH = SCREEN_WIDTH,
  SCREEN_HEIGHT = SCREEN_HEIGHT,
  sandbox = sandbox,
}
return mario_game
