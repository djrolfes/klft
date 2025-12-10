-- .nvim.lua at project root

-- 1) find/choose build dir
local function find_build_dir()
  if vim.env.CMAKE_BUILD_DIR and vim.env.CMAKE_BUILD_DIR ~= "" then
    return vim.env.CMAKE_BUILD_DIR
  end
  local d = vim.fn.finddir("build", ".;")
  return (d ~= "" and d) or "build"
end
local build = find_build_dir()

-- 2) detect multi-config from CMakeCache.txt (VS/Xcode/Ninja Multi-Config)
local function is_multi_config(builddir)
  local cache = builddir .. "/CMakeCache.txt"
  if vim.fn.filereadable(cache) == 0 then return false end
  for line in io.lines(cache) do
    if line:match("^CMAKE_GENERATOR:INTERNAL=.*Ninja Multi-Config")
       or line:match("^CMAKE_GENERATOR:INTERNAL=.*Visual Studio")
       or line:match("^CMAKE_GENERATOR:INTERNAL=.*Xcode") then
      return true
    end
  end
  return false
end

-- default config when multi-config; override with env or command below
vim.g.cmake_build_config = vim.env.CMAKE_BUILD_CONFIG or "Debug"

-- 3) set :make to cmake --build ... (adds --config if multi-config)
local function set_makeprg()
  if is_multi_config(build) then
    vim.opt_local.makeprg =
      ("cmake --build %s --parallel --config %s")
      :format(vim.fn.shellescape(build), vim.g.cmake_build_config)
  else
    vim.opt_local.makeprg =
      ("cmake --build %s --parallel")
      :format(vim.fn.shellescape(build))
  end
end
set_makeprg()

-- 4) convenience commands
vim.api.nvim_create_user_command("CMakeConfigure", function(opts)
  local arg = opts.args
  local cmd
  if vim.fn.filereadable("CMakePresets.json") == 1 and arg ~= "" then
    cmd = ("cmake --preset %s"):format(arg)
  else
    local bt = (arg ~= "" and arg) or "Debug"
    cmd = ("cmake -S . -B %s -DCMAKE_BUILD_TYPE=%s -DCMAKE_EXPORT_COMPILE_COMMANDS=ON")
      :format(vim.fn.shellescape(build), bt)
  end
  vim.cmd("!" .. cmd)
  set_makeprg() -- generator may have changed; refresh makeprg
end, { nargs = "?" })

vim.api.nvim_create_user_command("CMakeBuild", function(opts)
  local args = (opts.args ~= "" and (" --target " .. opts.args)) or ""
  vim.cmd("make" .. args)
end, { nargs = "?" })

vim.api.nvim_create_user_command("CMakeSetConfig", function(opts)
  vim.g.cmake_build_config = opts.args
  set_makeprg()
  print("CMake build config set to " .. vim.g.cmake_build_config)
end, {
  nargs = 1,
  complete = function() return { "Debug", "Release", "RelWithDebInfo", "MinSizeRel" } end,
})

-- Optional: open curses UI to tweak cache (your current workflow)
vim.api.nvim_create_user_command("CMakeCCMake", function()
  vim.cmd("!" .. ("ccmake %s"):format(vim.fn.shellescape(build)))
  set_makeprg()
end, {})
