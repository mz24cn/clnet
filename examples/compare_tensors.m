%used for compare two tensor data difference
function [delta max_e most l r] = compare_tensors(dir_l, dir_r, name, transpose, threshold)
    l = csvread([dir_l, name, '.csv'], 1, 0);
    r = csvread([dir_r, name, '.csv'], 1, 0);
    if exist('transpose', 'var') && transpose
        r = r';
    end
    if ~exist('threshold', 'var')
        threshold = 0.1;
    end
	delta = abs(l - r) ./ (mean(mean(abs(r))) + 1e-38);
    max_e = max(max(delta));
	[rows cols] = find(delta / max_e > threshold);
    most = zeros(length(rows), 5);
    for i = 1 : length(rows)
        most(i,1) = rows(i);
        most(i,2) = cols(i);
        most(i,3) = l(rows(i), cols(i));
        most(i,4) = r(rows(i), cols(i));
    end
    most(:,5) = abs(most(:,3) - most(:,4)) ./ (abs(most(:,4)) + 1e-38);
end