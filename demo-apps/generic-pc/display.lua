local display = {} -- main display object

local win = false

local colours = {
   'white', 'blue', 'green', 'darkGreen', 'orange', 'cyan',
   'darkCyan', 'magenta', 'darkMagenta', 'purple', 'brown',
   'gray', 'darkGray', 'red', 'darkRed', 'yellow', 'darkYellow',
}

local function prep_verbose(opt, source, class_names, target_list)
   local classes = class_names
   local targets = target_list
   local diw = opt.diw
   local threshold = opt.pdt
   local zoom = opt.zoom or 1
   local mean_idx = 0
   local top_n = opt.dv

   -- result integration tensors
   local fps_avg = torch.FloatTensor(diw):zero()
   local results_matrix = torch.FloatTensor(#classes, diw):zero()
   local results_mean = torch.FloatTensor(#classes):fill(-1)

   -- calculate display 'zoom' and text offsets
   local side = math.min(source.h, source.w)
   local win_w = zoom*side
   local win_h = zoom*side

   local update_results_mean = function(results, mean_idx)
      for i=1, #classes do
         results_matrix[i][mean_idx] = results[i] or 0
         results_mean[i] = torch.mean(results_matrix[i])
      end
   end

   local z = (zoom * side) / 512
   local box_w = 180*z
   local box_h = (60*z)+(20*z*top_n)
   local fps_x = 10*z
   local fps_y = 30*z
   local text_x1 = 10*z
   local text_x2 = 110*z
   local text_y = 20*z

   -- create window
   if not win then
      win = qtwidget.newwindow(win_w, win_h, 'TeraDeep Demo App')
   else
      win:resize(win_w, win_h)
   end

   -- Set font size to a visible dimension
   win:setfontsize(20*z)


   display.forward = function(results, img, fps)
      results = results or {}
      mean_idx = mean_idx + 1
      if mean_idx > diw then
         mean_idx = 1
      end
      fps_avg[mean_idx] = fps or 0

      -- rolling buffer of detection likelihoods
      update_results_mean(results, mean_idx)

      win:gbegin()
      win:showpage()

      -- displaying current frame
      image.display{image = img, win = win, zoom = zoom}

      -- rectangle for text:
      win:rectangle(0, 0, box_w, box_h)
      win:setcolor(0, 0, 0, 0.4)
      win:fill()

      -- report fps
      win:moveto(fps_x, fps_y)
      win:setcolor(1, 0.3, 0.3)
      win:show(string.format('%.2f fps', torch.mean(fps_avg)))

      win:setcolor(.3,.8,.3)
      _, idx = results_mean:sort(true)
      for i = 1, top_n  do
         local move_to_y = text_y*(2+i)
         win:moveto(text_x1, move_to_y)
         win:show(string.format('%s ', classes[idx[i]]))
         win:moveto(text_x2, move_to_y)
         win:show(string.format('(%3.0f%%)', results_mean[idx[i]]*100))
      end
      win:gend()
   end

   -- set screen grab function
   display.screen = function()
      return win:image()
   end

   -- set continue function
   display.continue = function()
      return win:valid()
   end

   -- set close function
   display.close = function()
      if win:valid() then
         win:close()
      end
   end
end


function display:init(opt, source, class_names, target_list, custom_colours)
   assert(opt and ('table' == type(opt)))
   assert(source and ('table' == type(source)))
   assert(class_names and ('table' == type(class_names)))
   require 'qtwidget'

   colours = custom_colours or colours
   local colours_length = #colours
   for i=1, (#class_names) do
      -- replicate colours for extra categories
      local index = (i == colours_length) and colours_length or i%colours_length
      colours[colours_length+i] = colours[index]
   end

   prep_verbose(opt, source, class_names, target_list)

end

return display
