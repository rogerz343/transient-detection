function [] = randomize_data_rows(filename)
% The data files right now have all the 0 examples on top and all the 1
% examples on the bottom; I'm not sure if this affects anything, so I made
% this file so you can randomize the order of the examples

file = load(filename);
file = file.dat;
nrows = size(file,1);
random_rows = transpose(randperm(nrows));
dat = file(random_rows,:);
save(filename, 'dat');

end