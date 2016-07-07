require "torch"

require "mario_uct_model"

local function mario_uct_model_main()
  local model = mario_uct_model.UctModel:new()
  
  model.save_to = "uct_model.sav"
  model.log_file = io.open("uct_model.log", "a")

  model.result_actions = {}
  model.all_actions = {}
  model.num_skip_frames = 12

  model.min_num_visits_to_expand_node = 1
  model.max_num_runs = 100
  model.max_depth = 60
  
  model.enable_debug = true

  model:main()
end

mario_uct_model_main()
