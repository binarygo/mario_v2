require "torch"

require "mario_uct_model"

local function mario_uct_model_main()
  local model = mario_uct_model.UctModel:new()
  model.save_to = "uct_model.sav"
  model.log_file = io.open("uct_model.log", "a")
  model.result_actions = {}
  model.all_actions = {
    0x00, 0x04, 0x01
  }
  model.min_num_visits_to_expand_node = 2
  model.max_num_runs = 5
  model.max_depth = 10
  model.enable_debug = true
  model:main()
end

mario_uct_model_main()
