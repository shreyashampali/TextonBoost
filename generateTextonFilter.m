function [filter_locs,filter_sizes] = generateTextonFilter(count,boundary,size)
    filter_locs = round(random('uniform',boundary(1),boundary(2),count,2));
    filter_sizes = round(random('uniform',size(1),size(2),count,2));
end