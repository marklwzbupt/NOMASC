function [n_output, f_output, time] = turbo_ldpc_modulate_noma_awgn(n_input, f_input, n_rou, f_rou, n_snr, f_snr)
% 调制阶数
M_n = 16;
M_f = 4;
% 定义LDPC 编解码器
coderate  = 1/2;
ldpcEncoder = comm.LDPCEncoder(dvbs2ldpc(coderate));
ldpcDecoder = comm.LDPCDecoder(dvbs2ldpc(coderate));
turboEnc = comm.TurboEncoder('InterleaverIndicesSource','Input port');
turboDec = comm.TurboDecoder('InterleaverIndicesSource','Input port','NumIterations',4);
n_input = transpose(str2num(int2str(cell2mat(n_input))));
f_input = transpose(str2num(int2str(cell2mat(f_input))));
n_snr = str2num(int2str(n_snr));
f_snr = str2num(int2str(f_snr));
% 定义输入数组, n_input是文本，使用turbo编码，f_input是图像，使用ldpc编码
n_len = length(n_input);
f_len = length(f_input);
intrlvrIndices = randperm(n_len);
% n_input使用turbo编码, f_input使用ldpc编码
turbo_encoded = turboEnc(n_input, intrlvrIndices);
f_input_1 = [f_input; randi([0 1], 64800 * coderate - f_len, 1)];
ldpc_encoded = ldpcEncoder(f_input_1);
turbo_encoded1 = [turbo_encoded ; randi([0 1], 64800 / log2(M_f) * log2(M_n) - length(turbo_encoded), 1)];

% 调制
n_mod = qammod(turbo_encoded1, M_n, 'InputType', 'bit', 'UnitAveragePower', true);
f_mod = qammod(ldpc_encoded, M_f, 'InputType', 'bit', 'UnitAveragePower', true);
% 叠加编码
sc_coding = sqrt(n_rou) .* n_mod + sqrt(f_rou) .* f_mod;
% 产生瑞利衰落向量
% hn = (randn(64800 / log2(M),1) + 1i * randn(64800 / log2(M),1)) / sqrt(2);
% hf = (randn(64800 / log2(M),1) + 1i * randn(64800 / log2(M),1)) / sqrt(2);
sigma_n = mean(sc_coding .* conj(sc_coding)) / (10 ^ (n_snr / 10));
sigma_f = mean(sc_coding .* conj(sc_coding)) / (10 ^ (f_snr / 10));
noise_n = sqrt(sigma_n) .* (randn(64800 / log2(M_f),1) + 1i * randn(64800 / log2(M_f),1)) / sqrt(2);
noise_f = sqrt(sigma_f) .* (randn(64800 / log2(M_f),1) + 1i * randn(64800 / log2(M_f),1)) / sqrt(2);
% 加入噪声
y_n = sc_coding + noise_n;
y_f = sc_coding + noise_f;

% sic 过程，f用户直接解码
% 均衡
tic;
var1 = mean(sc_coding .* conj(sc_coding)) / (10 ^ (f_snr / 10));
f_demod = qamdemod(y_f / sqrt(f_rou),M_f,'OutputType','llr', 'UnitAveragePower', true, 'NoiseVariance', (var1 + n_rou) / f_rou);
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
n_bit_esti = turboDec(-n_demod(1:length(turbo_encoded)),intrlvrIndices);

n_output = double(n_bit_esti);
f_output = double(f_bit_esti(1:f_len));
time = size(n_bit_esti);
n_error = biterr(n_input, n_output)/n_len
f_error = biterr(f_input, f_output)/f_len;

