require "torch"

require "mario_util"
require "mario_game"

local function mario_uct_replay_main()
  data = torch.load("uct_model.sav.model.2")
  
  local num_skip_frames = data.num_skip_frames
  local result_actions = data.result_actions
  mario_game.sandbox:startGame(nil)
  local save = mario_game.sandbox:saveGame()
  while true do
    mario_game.sandbox:startGame(save)
    for i, a in ipairs(result_actions) do
      mario_game.sandbox:advance(a, num_skip_frames)
    end
  end
end

mario_uct_replay_main()
