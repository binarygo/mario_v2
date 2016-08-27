require "torch"

require "mario_uct_model"

local function mario_uct_model_main()
  init_data = { result_actions = {}, num_skip_frames = 12 }
  -- init_data = torch.load("uct_model.sav.model.4")
  
  local model = mario_uct_model.UctModel:new()

  model.all_actions = {
    0x00, -- nil
    0x08, -- left
    0x04, -- right
    0x20, -- up
    0x10, -- down
    0x02, -- A
    0x01, -- B
    0x06, -- right + A
    0x05, -- right + B
  }

  model.game_save = nil
  model.result_actions = init_data.result_actions
  model.num_skip_frames = init_data.num_skip_frames
  
  model.min_num_visits_to_expand_node = 1
  model.max_num_runs = 120
  model.max_depth = 60
  model.mario_score_cell = 200
  model.use_ucb1 = true
  
  model.save_to = "uct_model.sav"
  model.log_file = io.open("uct_model.log", "a")
  model.enable_debug = true

  model:main()
end

mario_uct_model_main()
