function [n_output, f_output, time] = ldpc_modulate_noma_awgn(n_input, f_input, n_rou, f_rou, n_snr, f_snr)
% 调制阶数
M_n = 64;
M_f = 16;
% 定义LDPC 编解码器
coderate  = 1/2;
ldpcEncoder = comm.LDPCEncoder(dvbs2ldpc(coderate));
ldpcDecoder = comm.LDPCDecoder(dvbs2ldpc(coderate));
n_input = transpose(str2num(int2str(cell2mat(n_input))));
f_input = transpose(str2num(int2str(cell2mat(f_input))));
n_snr = str2num(int2str(n_snr));
f_snr = str2num(int2str(f_snr));
% 定义输入数组
n_len = length(n_input);
f_len = length(f_input);

% 定义输入数组
n_input_1 = [n_input;randi([0 1], 64800 * coderate - n_len, 1)];
f_input_1 = [f_input; randi([0 1], 64800 * coderate - f_len, 1)];

% LDPC编码
n_ldpc = ldpcEncoder(n_input_1);
f_ldpc = ldpcEncoder(f_input_1);

%补长度
n_ldpc = [n_ldpc; randi([0 1], 64800 / log2(M_f) * log2(M_n) - length(n_ldpc), 1)];

% 调制
n_mod = qammod(n_ldpc, M_n, 'InputType', 'bit', 'UnitAveragePower', true);
f_mod = qammod(f_ldpc, M_f, 'InputType', 'bit', 'UnitAveragePower', true);
% 叠加编码
sc_coding = sqrt(n_rou) .* n_mod + sqrt(f_rou) .* f_mod;
% 加入噪声
y_n = awgn(sc_coding, n_snr, 'measured');
y_f = awgn(sc_coding, f_snr, 'measured');
tic;
% sic 过程，f用户直接解码
var1 = mean(sc_coding .* conj(sc_coding)) / (10 ^ (f_snr / 10));
f_demod = qamdemod(y_f / sqrt(f_rou),M_f,'OutputType','llr', 'UnitAveragePower', true, 'NoiseVariance', (var1+n_rou) / f_rou);
f_bit_esti = ldpcDecoder(f_demod);

%mean(sc_coding .* conj(sc_coding))
% n用户sic解码
var2 = mean(sc_coding .* conj(sc_coding)) / (10 ^ (n_snr / 10));
f_n_demod = qamdemod(y_n / sqrt(f_rou), M_f, 'OutputType','llr', 'UnitAveragePower', true, 'NoiseVariance', (var2+n_rou) / f_rou );
f_n_bit = ldpcDecoder(f_n_demod);

f_n_bit_ldpc = ldpcEncoder(double(f_n_bit));
f_n_bit_remod = qammod(f_n_bit_ldpc, M_f, 'InputType', 'bit', 'UnitAveragePower', true);
var3 = mean(sc_coding .* conj(sc_coding)) / (10 ^ (log10(n_rou / mean(sc_coding .* conj(sc_coding))) + n_snr / 10));
residual = y_n - sqrt(f_rou).* f_n_bit_remod;
n_demod = qamdemod(residual./ sqrt(n_rou), M_n, 'OutputType','llr','UnitAveragePower', true,  'NoiseVariance', var2 ./ n_rou);
n_bit_esti = ldpcDecoder(n_demod(1:64800));

n_output = double(n_bit_esti(1:n_len));
f_output = double(f_bit_esti(1:f_len));
time = size(sc_coding);
n_error = biterr(n_input_1, n_bit_esti)/(64800 * coderate)
f_error = biterr(f_input_1, f_bit_esti)/(64800 * coderate)
end