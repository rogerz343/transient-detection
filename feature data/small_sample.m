function [] = small_sample(filename, nsamples, newfilename)
% Takes a small sample of a certain matrix, so that you can work with
% smaller data sets at first

randomize_data_rows(filename);
file = load(filename);
file = file.dat;
dat = file(1:nsamples,:);
save(newfilename, 'dat');

end