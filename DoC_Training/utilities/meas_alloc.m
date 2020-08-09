function meas = meas_alloc(subrate, img_size, ratio, levels)
test = 0;
if test == 1
    ratio       = 2; 
    subrate     = 0.1; 
    img_size    = 32;
    levels      = 2; 
end

weight = 1; 
for i = 1:1:levels-1
    weight(i+1) = i * ratio;
end

total_meas = round(subrate * img_size^2);
avg_meas = floor(total_meas/sum(weight)); 
for i = 1:1:levels-1
    meas(i) = avg_meas * weight(i); 
end
meas(i+1) = total_meas - sum(meas); 
meas = flip(meas);

