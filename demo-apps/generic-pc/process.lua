local process = {} -- process object

local function multiscale(img, network)
   assert(require('image'))
   local eye = opt.is
   local scaled = nil
   if img:dim() == 4 and img:size(1) == 1 then
      img = img:squeeze(1)
   end
   if img:dim() == 3 then
      -- no batch case
      scaled = image.scale(img,'^' .. eye)
      -- normalize the input:
      for c = 1,3 do
         scaled[c]:add(-network.stat.mean[c])
         scaled[c]:div( network.stat.std [c])
      end
   end
   return scaled
end

local function prep_spatial_0_cpu(opt, source, network)
   assert(require('image'))
   local eye = opt.is

   local side = math.min(source.h, source.w)
   local x1  = source.w / 2 - side / 2
   local y1  = source.h / 2 - side / 2
   local x2  = source.w / 2 + side / 2
   local y2  = source.h / 2 + side / 2
   local cropped = nil

   process.forward = function(img)
      cropped = image.crop(img, x1, y1, x2, y2)

      local scaled = multiscale(cropped, network)
      -- computing network output
      local results = network.model:forward(scaled)
      return results, cropped
   end
end

function process:init(opt, source, network)
   prep_spatial_0_cpu(opt, source, network)
end

return process
