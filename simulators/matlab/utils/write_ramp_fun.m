function write_ramp_fun(delay, amplitude, pulse_width, interphase_gap, decay_width, k, ramp_width, num)
% Load the template as a string
template_file = 'utils/template_pulse_exp_ramp.m';
template_content = fileread(template_file);

% Replace each placeholder
template_content = strrep(template_content, '{{delay}}', num2str(delay));
template_content = strrep(template_content, '{{amplitude}}', num2str(amplitude));
template_content = strrep(template_content, '{{pulse_width}}', num2str(pulse_width));
template_content = strrep(template_content, '{{interphase_gap}}', num2str(interphase_gap));
template_content = strrep(template_content, '{{decay_width}}', num2str(decay_width));
template_content = strrep(template_content, '{{k}}', num2str(k));
template_content = strrep(template_content, '{{ramp_width}}', num2str(ramp_width));

% Save the modified content to a new function file
currentDir = pwd;

fun_wrapper_filename = fullfile('utils/tmp_funs/', sprintf('temp_ramp_%s.m',num));

fid = fopen(fun_wrapper_filename, 'w');
if fid == -1
    error("Failed to open file: %s", fun_wrapper_filename)
else

    fwrite(fid, template_content);
    fclose(fid);
end

end
