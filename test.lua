require "torch"

local game_state = {}

local function DecodeJoypadInput(input_code)
  -- input_code is an 8-bit integer in [0, 255]
  -- start|select|A|B|up|down|left|right
  return {
    start = (AND(input_code, 0x80) ~= 0),
    select = (AND(input_code, 0x40) ~= 0),
    A = (AND(input_code, 0x20) ~= 0),
    B = (AND(input_code, 0x10) ~= 0),
    up = (AND(input_code, 0x08) ~= 0),
    down = (AND(input_code, 0x04) ~= 0),
    left = (AND(input_code, 0x02) ~= 0),
    right = (AND(input_code, 0x01) ~= 0)
  }
end

local function JoypadInputToString(input)
  ss = {""}
  for i, key in ipairs{
    "start", "select", "A", "B", "up", "down", "left", "right"} do
    ss[#ss + 1] = string.format("%s=%d", key, input[key] and 1 or 0)
  end
  ss[#ss + 1] = ""
  return table.concat(ss, "|")
end

local function SetJoypad(player, input_code)
  local input = DecodeJoypadInput(input_code)
  joypad.set(player, input)
  return input
end

local function GetState()
  return {
    die = memory.readbyte(0x0747),
    lives_screen = memory.readbyte(0x0757),
    time_up = memory.readbyte(0x0759),
    lives = memory.readbyte(0x075A),
    world = memory.readbyte(0x075F),
    level = memory.readbyte(0x0760),
    game_mode = memory.readbyte(0x0770),
    game_status = memory.readbyte(0x0772),
    score = {
      memory.readbyte(0x07DD),
      memory.readbyte(0x07DE),
      memory.readbyte(0x07DF),
      memory.readbyte(0x07E0),
      memory.readbyte(0x07E1),
      memory.readbyte(0x07E2),
    }
  }
end

local function GetGameMode()
  return memory.readbyte(0x0770)
end

local function GetGameStatus()
  return memory.readbyte(0x0772)
end

local function GetPlayerState()
  return memory.readbyte(0x000E)
end

local function GetLiveScreenFlag()
  return memory.readbyte(0x0757)
end

local function GetMarioScore()
  local score = 0
  local score_addr = 0x07DD
  for i = 1,6 do
    score = score * 10 + memory.readbyte(score_addr)
    score_addr = score_addr + 1
  end
  return score
end

local function GetMarioLives()
  return memory.readbyte(0x075A)
end

local function GetWorldLevel()
  return memory.readword(0x075F)
end
  
local function SkipLiveScreen()
  while GetLiveScreenFlag() == 0 do
    emu.frameadvance()
  end
  while GetLiveScreenFlag() == 1 do
    emu.frameadvance()
  end
end

local function UpdateGameState()
  game_state.world_level = GetWorldLevel()
  game_state.mario_lives = GetMarioLives()

  local current_h = memory.readbyte(0x0778)
  if game_state.h then
    game_state.sum_h = game_state.sum_h + math.max(0, current_h - game_state.h)
  else
    game_state.sum_h = 0
  end
  game_state.h = current_h

  local x_block = memory.readbyte(0x006D)
  if x_block ~= game_state.x_pre_block and x_block ~= game_state.x_block then
    game_state.x_pre_block = game_state.x_block
    game_state.x_block = x_block
    game_state.x_no = game_state.x_no + 1
  elseif x_block == game_state.x_pre_block then
    game_state.x_pre_block = nil
    game_state.x_block = x_block
    game_state.x_no = game_state.x_no - 1
  end
  
  local x = memory.readbyte(0x0086)
  game_state.x = x
end

local function StartGame()
  emu.speedmode("normal")
  emu.message("")
  for i = 1, 100 do
    emu.frameadvance()
  end
  SetJoypad(1, 0x80)
  SkipLiveScreen()
  game_state.acc_x = 0
  game_state.x_pre_block = nil
  game_state.x_block = memory.readbyte(0x006D)
  game_state.x_no = 0
  game_state.x = memory.readbyte(0x0086)
end

local function SaveGame()
  local save = savestate.object()
  savestate.save(save)
  game_state.save = save
end

local function LoadGame()
  if game_state.save then
    savestate.load(game_state.save)
  end
end

local function IsLevelEnd()
  return GetWorldLevel() ~= game_state.world_level
end

local function IsMarioDies()
  return GetPlayerState() == 11
end

local function Main()
  StartGame()
  SaveGame()

  while true do
    emu.message("Game starts!")
    UpdateGameState()
    while true do
      -- todo: populate buff
      emu.frameadvance()
      -- set time to 999
      --memory.writebyte(0x07F8, 9)
      --memory.writebyte(0x07F9, 9)
      --memory.writebyte(0x07FA, 9)
      emu.message(string.format(
        "w=%d, l=%d, b=%d, x=%d",
        memory.readbyte(0x075F),
        memory.readbyte(0x0760),
        memory.readbyte(0x006D),
        memory.readbyte(0x0086)))
      --if IsMarioDies() then
      --  emu.message("Mario dies!")
      --  break
      --end
      --if IsLevelEnd() then
      --  emu.message("Level clear!")
      --  break
      --end
      if memory.readbyte(0x06D7) > 0 then
        break
      end
      UpdateGameState()
    end
  end
end

function Test()
  StartGame()
  for i = 1, 1000 do
    emu.frameadvance()
  end
  screen = torch.ByteTensor(torch.ByteStorage():string(gui.gdscreenshot()))
  f = torch.DiskFile("screen.data", "w")
  f:writeObject(screen)
  f:close()
end

Main()
