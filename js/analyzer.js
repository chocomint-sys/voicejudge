/*
 * VoiceAnalyzer — 音声のスペクトラム解析と地声／裏声の推定
 *
 * 処理の流れ
 *   1. STFT（ハン窓）でパワースペクトログラムを計算
 *   2. McLeod Pitch Method（NSDF）でフレームごとの基本周波数 F0 を推定
 *   3. 有声フレームごとに、倍音構造と周期性の特徴量を求める
 *        - 倍音どうしのレベル差: H1−H2, H1*−H2*（フォルマント補正）, H1−H4, H2−H4, H1−A3 など
 *        - 高域の強さ: 倍音豊富度（HRF）, スペクトル傾斜, alpha ratio, 高域/低域比, スペクトル重心 / F0
 *        - 周期性の明瞭さ: CPP（ケプストラムピークの突出度）, 倍音と雑音の比, 分数倍音比
 *   4. 実際の歌声（ccmusic chest_falsetto・GTSinger。通話モードを模擬した音声を含む）で学習した
 *      モデル（REGISTER_MODEL）でフレームごとに裏声スコアを求め、平滑化して多数決で全体を判定
 *
 * ブラウザでは window.VoiceAnalyzer、Node.js では module.exports として使える。
 */
(function (root) {
  'use strict';

  const DEFAULTS = {
    frameSize: 2048,        // STFT の窓長（22.05 kHz で約 93 ms）
    hopSize: 256,           // フレーム間隔（約 11.6 ms）
    pitchFrameSize: 1024,   // ピッチ推定の窓長
    minF0: 60,
    maxF0: 1400,
    clarityThreshold: 0.75, // NSDF ピーク値がこれ未満なら無声
    relativeSilenceDb: -35, // 最大フレームからこれ以上小さいフレームは無声
    absoluteSilenceDb: -60, // dBFS
    harmonicMaxFreq: 4000,  // 特徴量に使う倍音の上限周波数
    minVoicedRun: 5,        // これより短い有声区間はノイズとして除外（フレーム数）
    scoreSmoothing: 9,      // スコアの移動中央値の幅（フレーム数）
  };

  // 地声／裏声の判定モデル（実際の歌声データで学習した小さなニューラルネット）
  //   features の順に特徴量を並べ、欠損は medians で補完 → (x − mean) / scale → 隠れ層（ReLU）→ 出力（ロジスティック）
  //   threshold を超える確率なら裏声。chips は各特徴量単独での地声／裏声の境目（表示用）
  /* REGISTER_MODEL:BEGIN */
  // 学習データ: ccmusic chest_falsetto + GTSinger 20 名（通常音声＋通話モード模擬 2 種）。歌手単位の交差検証で balanced accuracy 81.5%（しきい値 0.56、地声 85.9%・裏声 77.0%）
  const REGISTER_MODEL = {"features":["h1h2","h1h2c","hrf","slope","h1h4","h2h4","shr","h1a3","hiLo","h1max","extent","hnr","alpha","centroidF0","cpp","highBandDb"],"medians":[4.90451,6.02396,-1.61615,-11.5068,22.4003,16.2658,-37.1922,23.7854,-23.2761,0,3.4873,39.6353,-13.9293,1.68149,20.5321,-30.93],"mean":[5.50384,5.89319,-1.55553,-11.5998,21.3286,15.8247,-36.7507,23.3926,-23.4965,-2.70136,3.36849,39.2076,-14.3698,2.08121,20.9196,-32.845],"scale":[10.724,6.13374,10.4425,4.12514,15.3152,11.6978,8.14363,14.1361,11.2484,4.74312,0.956262,10.3151,9.86198,1.35633,3.76695,21.1204],"threshold":0.56,"type":"mlp","layers":[{"W":[[0.019981,-0.203425,-0.194213,-0.0741931,-0.12676,0.531227,-0.242004,0.200212,0.633305,-0.206487,0.177552,-0.392562,-0.156906,0.569956,-0.975477,0.318165,-0.419325,0.199035,0.115474,0.975064,0.00742078,0.13876,-0.419891,0.627662,0.118148,0.148276,-0.267267,0.747907,0.0339536,-0.17542,0.4206,0.474408],[-0.356723,0.0772597,-0.334493,0.0295524,0.0478169,0.660259,0.172652,0.582528,-0.09293,0.261229,0.303151,-0.315009,-0.0191154,0.356038,-0.244523,-0.876389,0.0845569,0.324109,-0.142692,0.0165731,0.186072,-0.388467,0.0323095,0.156925,0.0503935,-0.0770356,0.0011419,-0.127678,-0.574286,-0.346185,0.188082,-0.0209555],[0.00461243,-0.338538,0.276948,-0.26759,-0.140426,0.248985,0.209004,-0.285002,0.15488,-0.0339699,0.421555,-0.148486,-0.935541,-0.724607,-0.510931,-0.657528,0.064228,-0.305404,-0.73346,0.0246498,0.188373,0.133183,-0.565494,-0.0479411,0.252995,-0.0274943,-0.554398,1.05218,0.282647,0.0576797,0.0406033,0.287869],[0.283162,-0.242246,0.405087,0.0457771,0.345976,-0.0787676,-0.124829,0.377241,-0.354869,0.321389,-0.260421,-0.305616,-0.249674,0.0166608,-0.100253,0.0887525,0.400611,0.137171,0.337408,-0.1872,-0.0345894,-0.20685,0.223861,0.869762,-0.36211,0.328967,-0.237914,-0.135549,-0.284987,0.0216395,0.150937,-0.0795795],[0.0768969,-0.0988321,-0.281635,-0.236862,-0.423501,-0.415084,-0.0937631,-0.053547,0.178457,-0.00350975,-0.127055,-0.388009,0.0085155,0.26906,-0.257124,0.796242,0.115558,0.331239,-0.305433,1.02412,0.340469,0.186737,-0.410194,0.872632,0.154968,0.0826666,-0.312136,0.370137,0.378242,-0.0975715,0.2562,-0.0644668],[0.0899642,0.417609,-0.0614936,0.11153,0.188642,-0.386843,-0.343764,-0.177671,-0.0573637,-0.272917,-0.224786,-0.32497,-0.317435,-0.129349,0.500784,-0.134007,-0.203676,0.625301,0.212789,-0.115502,0.256389,-0.0507578,0.225025,0.241687,0.46548,0.0113847,-0.0623794,-0.197994,0.369275,-0.212186,-0.28292,-0.176802],[-0.529042,-0.0059838,0.0326118,0.197706,-0.223828,-0.45281,0.00220389,0.036908,-0.00201174,-0.0935445,-0.214049,-0.440249,-0.240459,-0.275225,-0.165143,-0.00989147,-0.1094,0.144703,0.635433,0.287571,0.422094,-0.197,-0.300278,-0.250366,-0.682055,0.369785,0.0164813,-0.0347396,-0.326578,0.386186,0.0228023,0.119104],[-0.142411,1.02233,0.819603,-0.40892,0.29545,0.560828,-0.218959,-0.0479011,-0.534281,-0.677412,-0.348123,-0.253726,0.104935,0.582539,0.0248907,-0.276377,0.492944,-0.303311,0.0845525,-0.0124575,-0.0636592,-0.481381,0.544054,0.233782,-0.224047,-0.0357553,-0.586513,0.398177,0.522612,0.512461,0.553361,-0.831956],[0.0366836,-0.169922,0.792215,-0.523471,-0.484436,-0.762857,-0.736337,0.648202,-0.0607379,-0.0156547,0.265291,0.337795,0.0449617,0.145095,1.19528,0.353241,-0.0839701,0.289569,-0.40345,0.423055,0.757495,0.257449,0.231915,0.836407,-0.363725,0.146267,1.1574,-0.177507,-0.207887,-0.625165,-0.0534285,0.142181],[-0.644058,-0.239878,0.438259,0.318562,0.224867,-0.771216,-0.159969,-0.0938683,-0.0993605,0.48774,0.260169,0.287127,-0.110674,0.281229,0.660644,-0.392251,0.147362,-0.472535,-0.310231,-0.982632,0.166802,-0.592126,-0.100476,-0.322979,0.267468,-0.46587,0.00306807,-0.348666,0.163187,-0.12241,-0.290767,-0.109281],[-0.181474,-0.11348,-0.246838,0.0589693,-0.252649,-0.287684,0.0463931,0.836988,0.0926233,0.370395,-0.332542,0.00188382,-0.0725159,0.307794,0.0743904,-0.0327233,-0.195296,-0.345463,0.0799067,0.0200333,0.263264,-0.428614,0.0505734,-0.653666,0.073256,-0.410478,0.10634,0.0124822,0.222975,0.219335,0.373174,-0.344681],[-0.194443,0.0558587,-0.0762187,1.00282,-0.351977,-0.64666,-0.214812,0.0576098,-0.160456,0.11299,0.393703,0.0705518,-0.271467,-0.304079,0.0530052,-0.137221,0.00390775,0.50616,-0.218224,0.231397,0.702641,0.361494,-0.59706,0.119403,0.532332,1.11479,0.105579,-0.177651,0.0943099,0.231989,0.999868,0.264652],[0.247981,0.992485,-0.210447,-0.521087,-0.0621345,-0.581858,-0.220675,0.223606,-0.0536041,0.439574,-0.889306,0.799071,-0.345519,-0.114937,-0.73888,0.229265,0.153399,0.733806,-0.218615,0.0432036,0.490686,-0.515266,0.00221184,-0.900923,0.677584,-0.471794,-0.363523,-0.333121,-0.300393,-0.366476,0.000867588,-0.495989],[0.184036,-0.8803,-0.036201,0.581233,0.62783,1.40481,0.209822,-0.823089,-0.248926,-0.575051,-0.553783,-0.258251,-0.452336,0.0304401,0.622283,0.799665,-0.0621715,-0.0925476,-0.142097,0.217641,0.284214,-0.582122,0.292784,1.17559,0.573655,0.268815,1.0212,-0.816765,-0.0376642,0.0437607,0.555867,1.35783],[0.293832,0.316176,-0.166857,0.117479,-0.110944,-0.0338817,0.456045,-0.092543,0.0735571,0.0183462,0.405877,0.632261,-0.141186,0.339852,0.148425,-0.0235128,-0.16229,0.112173,0.110486,-0.125875,0.28698,-0.0020445,0.0795727,-0.075042,0.0633099,-0.0938882,0.441037,0.121627,0.0577437,0.4494,-0.167741,-0.260218],[-0.119001,-0.180255,-0.952958,-0.00558976,-1.18345,0.172822,-0.393003,0.052337,0.816904,-0.958658,0.341835,-0.716203,0.368337,-0.368502,-0.228216,0.201318,-1.22091,-0.0800581,-0.444963,0.00814719,-0.743831,-1.46587,-0.112827,0.19475,0.309202,0.734654,1.16791,0.142018,-0.707616,0.368563,-0.0461784,-0.752894]],"b":[0.387541,-0.408203,0.461137,0.594382,-0.534317,-0.514758,0.729112,0.325731,-0.384208,-0.0850318,-0.735791,-0.0319931,-0.678292,0.0209911,-0.181814,0.175678,0.0348148,0.430036,-0.0864229,-0.0413636,0.0380889,0.0467067,-0.168179,-0.932735,-0.0726515,-0.058098,-0.821277,0.64358,-0.318639,0.0757648,0.221243,0.453396]},{"W":[[0.467244,-0.169621,-0.13659,0.625485,-0.107212,-0.468271,-0.260202,-0.146901,-0.0181337,0.0353366,-0.449952,-0.0390758,-0.163137,0.0876403,0.289463,-0.151707],[-0.213227,-0.579187,0.192067,0.560619,0.356191,0.663566,-0.0659717,0.318428,-0.565937,-0.199828,-0.0387796,-0.84939,0.449228,-0.0470249,0.621994,-0.763622],[-0.438528,0.224918,0.124703,0.0977156,-0.528652,0.299643,-0.54131,-0.998617,-0.51094,-0.323603,-0.320915,-0.17758,-0.361494,0.504801,0.279064,-1.64758],[-0.00735357,0.584262,-0.27258,0.821714,-0.524449,-0.35793,0.67592,0.565723,-0.194933,0.266177,-0.13917,-0.62817,-0.171106,0.456303,-0.36202,0.0930309],[0.285002,0.346557,-0.0660098,0.576471,0.637665,-0.780428,-1.15762,-0.640928,0.443865,-0.979817,0.2889,0.705111,0.307395,-0.487785,0.627358,0.574331],[0.0431282,-0.0563407,0.701453,0.164039,-0.298259,-0.562725,-1.38161,0.31761,-0.187126,0.711311,0.284131,1.00296,-0.531782,0.0770412,0.19566,0.288199],[0.299684,0.843731,0.209393,-0.23389,0.261982,-0.326479,0.366996,0.490327,0.356462,-0.292801,-0.0300837,0.388761,0.18982,0.0456868,0.162529,-0.492508],[0.277938,-0.143598,-0.285091,0.449219,-0.0659831,0.462099,-0.45263,-0.413278,-0.107865,-0.631631,-0.396225,-0.173441,0.487579,0.225859,0.275343,0.138443],[0.395739,0.62158,0.664854,0.249408,0.137754,-0.324714,-1.05052,0.738693,-0.268369,-1.53229,0.111448,-0.154198,-0.208921,0.00805718,0.083691,-0.522051],[0.532322,-0.794483,0.655626,-0.654253,0.647177,0.118533,-0.0117117,0.30711,0.317505,-0.35483,-0.519797,0.219389,-0.119643,-0.650883,-0.472868,0.141339],[-0.376474,-1.03593,-0.442266,-0.15977,-0.210403,0.320057,-0.112919,0.313352,0.841608,0.178083,0.178602,-0.218257,-0.870726,0.503502,-0.894434,0.115653],[0.25979,0.306566,-0.283967,0.276109,-0.188797,0.460202,0.397225,0.331545,-0.105918,-0.125149,-0.409091,-0.174535,-0.19031,0.144889,0.438128,0.726592],[-0.0508815,-0.386515,0.0339825,-0.206805,-0.568588,-0.238622,0.252548,0.248048,0.264827,0.329248,0.377141,-0.00028681,0.760987,-0.155369,-0.322996,0.557879],[0.726662,0.0267857,0.157199,0.0172539,0.187666,-0.279953,-0.75715,-2.32486,0.176381,0.0311549,0.030247,0.178985,0.33941,0.547912,0.497098,-0.661193],[0.0507544,0.319686,-0.56102,0.696796,-0.994557,-0.449545,-0.196829,-0.409781,-0.960998,0.0285895,-0.321401,0.515795,-0.551645,-1.71113,0.958329,0.553064],[-0.288231,-0.394571,-0.0373555,-0.755353,0.494302,-0.432292,0.0789486,-0.924122,-0.155378,0.0120176,0.330771,0.319612,-0.24928,-0.494756,-0.484828,0.371343],[-0.623585,-0.254431,0.500523,-0.387109,0.12016,0.494508,0.835046,-0.917828,0.593439,0.489942,0.127893,0.127374,0.011453,-0.158958,-0.667524,1.17873],[-0.241459,0.396282,0.155395,0.0347067,-0.133457,-0.69795,-0.18076,-0.0829311,-0.524156,0.562735,-0.766049,0.488657,0.3915,-0.0779505,-0.0209555,-0.169486],[-0.18426,-0.323086,0.353436,0.0484113,0.024371,-0.295497,-0.21097,0.603835,-0.286718,0.071783,-0.387323,-0.0716052,-0.327502,0.239412,0.74852,-0.696824],[0.324747,0.025328,-0.518878,0.935958,-0.60948,0.568186,0.58652,-2.05921,-0.959694,-0.305837,-0.186554,-0.0968131,0.275432,0.47597,0.228485,0.4066],[0.382811,0.48444,-0.582842,-0.314206,0.709388,0.418639,0.392564,0.06159,0.471385,0.308374,0.338729,0.225704,-0.147806,0.208844,0.164852,-0.119221],[0.167079,-0.515445,0.459169,0.414739,0.381417,0.0981135,-0.491771,-1.0076,-0.0324221,0.799843,-0.0502018,-0.413262,0.118848,-0.714836,-0.164113,-0.663668],[0.199922,0.722176,0.559832,0.134345,-0.224702,-0.933053,-0.85987,-0.819329,0.560134,-0.58732,0.316249,0.207365,0.128814,-0.870735,0.993159,-0.70444],[0.828609,0.0466731,-0.14473,0.277531,-0.00294941,0.071909,0.566683,-0.520142,0.391734,0.0856368,0.105624,0.579855,0.462368,0.970347,0.919758,-0.322155],[0.0788022,-0.047898,0.43742,-0.683091,0.635568,-0.172699,-0.251535,-0.256727,-0.199098,-0.259572,0.208888,0.65989,-0.17753,0.279219,-0.301756,0.184026],[-0.141575,0.0764587,0.409044,-0.149342,-0.166006,-0.374462,0.0175682,0.332075,-0.31548,-1.03834,0.758562,-0.0722758,0.0155729,-0.466669,0.415798,0.711215],[-0.315892,-0.372923,-1.21961,-0.232454,-0.0751268,-0.132731,0.3216,-0.527096,0.137346,-0.0801713,1.62232,-0.748419,0.4368,0.0453267,0.266683,-0.249385],[0.0355329,-0.895359,-0.317706,-0.170113,0.217525,0.776858,0.570973,0.226403,0.142799,-0.577781,0.578702,0.21461,0.0459972,-0.170586,-0.531857,0.0416556],[-0.336874,0.26703,0.190337,-0.722556,-0.0416327,0.271416,-0.0382535,0.737802,0.225247,-0.08353,-0.237898,0.117079,-0.712803,0.375129,-0.316312,-0.453599],[0.202416,0.45734,-0.0829714,0.0209386,-0.154481,-0.0815106,-0.860901,-0.680674,-0.408276,0.0646372,-0.111145,0.0703831,-0.112481,-0.221277,0.175955,0.41939],[0.193999,0.55138,0.306412,-0.671009,0.778181,0.145132,0.338219,-0.358057,0.84999,-0.275875,0.237908,-0.245839,0.132203,-0.277015,-0.209132,-0.229443],[-0.497012,-0.417544,0.00618536,-0.0370123,-0.308941,-0.0895364,0.235679,0.992558,-0.409621,0.378565,-1.34738,-0.793836,-0.386785,-0.607176,0.400995,-0.548254]],"b":[-0.0140303,0.0688127,-0.263726,-0.177325,0.0614678,0.314509,0.0898601,-0.216398,0.737975,-0.815088,-0.415982,0.290982,0.902897,-0.275629,-0.138899,0.872912]},{"W":[[-0.717715],[-1.54555],[0.66112],[-1.77666],[-1.2029],[1.63925],[1.25701],[-1.96792],[-0.946674],[1.56318],[-1.65982],[1.28945],[1.44],[1.33597],[-1.37488],[1.92347]],"b":[0.320213]}],"chips":{"cpp":{"threshold":20.99,"dir":-1},"hrf":{"threshold":1.24,"dir":-1}}};
  /* REGISTER_MODEL:END */
  const NARROWBAND_DB = -45; // 4.6〜7.5 kHz と 2.6〜3.4 kHz の平均パワー比がこれ未満なら通話用の狭帯域入力とみなす

  // registerFeatures が返す特徴量（実際の歌声データで地声／裏声の分離に効いたもの）
  const EXTRA_FEATURES = ['h1h4', 'h2h4', 'shr', 'h1a3', 'hiLo', 'h1max', 'extent', 'hnr', 'alpha', 'centroidF0', 'cpp'];

  // ---------------------------------------------------------------- FFT

  function createFFT(n) {
    const levels = Math.round(Math.log2(n));
    if (1 << levels !== n) throw new Error('FFT size must be a power of 2');
    const cosT = new Float64Array(n / 2);
    const sinT = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) {
      cosT[i] = Math.cos((2 * Math.PI * i) / n);
      sinT[i] = Math.sin((2 * Math.PI * i) / n);
    }
    const rev = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      let r = 0;
      for (let b = 0, v = i; b < levels; b++, v >>= 1) r = (r << 1) | (v & 1);
      rev[i] = r;
    }
    return function transform(re, im) {
      for (let i = 0; i < n; i++) {
        const j = rev[i];
        if (j > i) {
          let t = re[i]; re[i] = re[j]; re[j] = t;
          t = im[i]; im[i] = im[j]; im[j] = t;
        }
      }
      for (let size = 2; size <= n; size *= 2) {
        const half = size / 2;
        const step = n / size;
        for (let i = 0; i < n; i += size) {
          for (let j = i, k = 0; j < i + half; j++, k += step) {
            const l = j + half;
            const tre = re[l] * cosT[k] + im[l] * sinT[k];
            const tim = -re[l] * sinT[k] + im[l] * cosT[k];
            re[l] = re[j] - tre;
            im[l] = im[j] - tim;
            re[j] += tre;
            im[j] += tim;
          }
        }
      }
    };
  }

  // ---------------------------------------------------------------- utilities

  function removeDC(input, sampleRate) {
    // 1 次の DC ブロッカー（カットオフ約 20 Hz）
    const R = Math.exp((-2 * Math.PI * 20) / sampleRate);
    const out = new Float32Array(input.length);
    let prevX = 0;
    let prevY = 0;
    for (let i = 0; i < input.length; i++) {
      const y = input[i] - prevX + R * prevY;
      prevX = input[i];
      prevY = y;
      out[i] = y;
    }
    return out;
  }

  function median(values) {
    if (values.length === 0) return NaN;
    const s = Array.from(values).sort((a, b) => a - b);
    const m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  }

  const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

  function noteName(freq) {
    if (!(freq > 0)) return '';
    const midi = Math.round(69 + 12 * Math.log2(freq / 440));
    return NOTE_NAMES[((midi % 12) + 12) % 12] + (Math.floor(midi / 12) - 1);
  }

  // ---------------------------------------------------------------- pitch (MPM)

  function createPitchDetector(sampleRate, o) {
    const W = o.pitchFrameSize;
    const M = W * 2;
    const fft = createFFT(M);
    const re = new Float64Array(M);
    const im = new Float64Array(M);
    const nsdf = new Float64Array(W);
    const minLag = Math.max(2, Math.floor(sampleRate / o.maxF0));
    const maxLag = Math.min(W - 2, Math.ceil(sampleRate / o.minF0));

    // frame: 長さ W の Float64Array。{ f0, clarity } を返す（検出不可なら f0 = 0）
    return function detect(frame) {
      let energy = 0;
      for (let n = 0; n < W; n++) {
        re[n] = frame[n];
        energy += frame[n] * frame[n];
      }
      re.fill(0, W);
      im.fill(0);
      if (energy <= 0) return { f0: 0, clarity: 0 };

      // Wiener–Khinchin: |X|^2 を FFT すると自己相関 × M が得られる
      fft(re, im);
      for (let k = 0; k < M; k++) {
        re[k] = re[k] * re[k] + im[k] * im[k];
        im[k] = 0;
      }
      fft(re, im);

      let m = 2 * energy;
      nsdf[0] = 1;
      for (let tau = 1; tau <= maxLag + 1; tau++) {
        const a = frame[tau - 1];
        const b = frame[W - tau];
        m -= a * a + b * b;
        nsdf[tau] = m > 1e-12 ? (2 * (re[tau] / M)) / m : 0;
      }

      // 正の区間ごとの最大値（key maxima）を集める
      const peaks = [];
      let tau = 1;
      while (tau <= maxLag && nsdf[tau] > 0) tau++;
      while (tau <= maxLag) {
        while (tau <= maxLag && nsdf[tau] <= 0) tau++;
        let best = -1;
        let bestTau = -1;
        while (tau <= maxLag && nsdf[tau] > 0) {
          if (nsdf[tau] > best) {
            best = nsdf[tau];
            bestTau = tau;
          }
          tau++;
        }
        if (bestTau >= minLag && bestTau < maxLag) peaks.push(bestTau);
      }
      if (peaks.length === 0) return { f0: 0, clarity: 0 };

      let highest = 0;
      for (const p of peaks) highest = Math.max(highest, nsdf[p]);
      const chosen = peaks.find((p) => nsdf[p] >= 0.9 * highest);

      const a = nsdf[chosen - 1];
      const b = nsdf[chosen];
      const c = nsdf[chosen + 1];
      const denom = a - 2 * b + c;
      const delta = denom !== 0 ? (0.5 * (a - c)) / denom : 0;
      const period = chosen + delta;
      return { f0: sampleRate / period, clarity: b - 0.25 * (a - c) * delta };
    };
  }

  // ---------------------------------------------------------------- formants (selective LPC)

  function levinson(R, order) {
    const a = new Float64Array(order + 1);
    const prev = new Float64Array(order + 1);
    a[0] = 1;
    let err = R[0];
    for (let i = 1; i <= order; i++) {
      let acc = R[i];
      for (let j = 1; j < i; j++) acc += a[j] * R[i - j];
      const k = -acc / err;
      prev.set(a);
      for (let j = 1; j < i; j++) a[j] = prev[j] + k * prev[i - j];
      a[i] = k;
      err *= 1 - k * k;
      if (!(err > 0)) break;
    }
    return a;
  }

  // z^p + c[1] z^(p-1) + ... + c[p] の根（Durand–Kerner 法）
  function polyRoots(c) {
    const p = c.length - 1;
    const re = new Float64Array(p);
    const im = new Float64Array(p);
    for (let i = 0; i < p; i++) {
      const ang = (2 * Math.PI * i) / p + 0.4;
      re[i] = 0.9 * Math.cos(ang);
      im[i] = 0.9 * Math.sin(ang);
    }
    for (let iter = 0; iter < 300; iter++) {
      let maxDelta = 0;
      for (let i = 0; i < p; i++) {
        const zr = re[i];
        const zi = im[i];
        let pr = 1;
        let pi = 0;
        for (let k = 1; k <= p; k++) {
          const nr = pr * zr - pi * zi + c[k];
          pi = pr * zi + pi * zr;
          pr = nr;
        }
        let dr = 1;
        let di = 0;
        for (let j = 0; j < p; j++) {
          if (j === i) continue;
          const ar = zr - re[j];
          const ai = zi - im[j];
          const nr = dr * ar - di * ai;
          di = dr * ai + di * ar;
          dr = nr;
        }
        const den = dr * dr + di * di || 1e-30;
        const qr = (pr * dr + pi * di) / den;
        const qi = (pi * dr - pr * di) / den;
        re[i] -= qr;
        im[i] -= qi;
        maxDelta = Math.max(maxDelta, Math.hypot(qr, qi));
      }
      if (maxDelta < 1e-12) break;
    }
    return { re, im };
  }

  // 0〜5.5 kHz のパワースペクトルに LPC を当てはめ、フォルマント [周波数, 帯域幅] を推定する
  function estimateFormants(power, offset, nBins, df) {
    const order = 12;
    const preEmphasis = 0.9;
    const B = Math.min(nBins - 1, Math.floor(5500 / df));
    const fs = 2 * B * df;
    const R = new Float64Array(order + 1);
    for (let b = 0; b <= B; b++) {
      const w = (Math.PI * b) / B;
      const p =
        power[offset + b] *
        (1 + preEmphasis * preEmphasis - 2 * preEmphasis * Math.cos(w)) *
        (b === 0 || b === B ? 0.5 : 1);
      for (let t = 0; t <= order; t++) R[t] += p * Math.cos(w * t);
    }
    R[0] *= 1.0001;
    const { re, im } = polyRoots(levinson(R, order));
    const formants = [];
    for (let i = 0; i < re.length; i++) {
      if (im[i] <= 0) continue;
      const freq = (Math.atan2(im[i], re[i]) * fs) / (2 * Math.PI);
      const bandwidth = (-Math.log(Math.hypot(re[i], im[i])) * fs) / Math.PI;
      if (freq >= 200 && freq <= 5000 && bandwidth > 0 && bandwidth < 700) formants.push([freq, bandwidth]);
    }
    formants.sort((x, y) => x[0] - y[0]);
    return { formants, fs };
  }

  // 下位 4 つのフォルマント共振が周波数 f に与えるゲイン [dB]（DC で 0 dB）
  function tractGainDb(formants, f, fs) {
    let gain = 0;
    for (let i = 0; i < Math.min(4, formants.length); i++) {
      const F = formants[i][0];
      const r = Math.exp((-Math.PI * Math.max(40, Math.min(500, formants[i][1]))) / fs);
      const c = Math.cos((2 * Math.PI * F) / fs);
      const num = (1 - 2 * r * c + r * r) ** 2;
      const den =
        (1 - 2 * r * Math.cos((2 * Math.PI * (F + f)) / fs) + r * r) *
        (1 - 2 * r * Math.cos((2 * Math.PI * (F - f)) / fs) + r * r);
      gain += 10 * Math.log10(num / den);
    }
    return gain;
  }

  // ---------------------------------------------------------------- harmonic features

  function harmonicFeatures(power, offset, nBins, df, f0, maxFreq) {
    const nyquist = df * (nBins - 1);
    const K = Math.floor(Math.min(maxFreq, nyquist - f0 / 2) / f0);
    if (K < 2) return null;

    const levels = new Float64Array(K + 1);
    const binCount = new Float64Array(K + 1);
    const valleys = [];
    const halfWidth = Math.max(1, Math.round((0.15 * f0) / df));
    for (let k = 1; k <= K; k++) {
      let lo = Math.max(1, Math.ceil(((k - 0.5) * f0) / df));
      let hi = Math.min(nBins - 1, Math.floor(((k + 0.5) * f0) / df));
      if (hi < lo) lo = hi = Math.round((k * f0) / df);
      let sum = 0;
      for (let b = lo; b <= hi; b++) sum += power[offset + b];
      levels[k] = Math.max(sum, 1e-20);
      binCount[k] = hi - lo + 1;
      // 倍音と倍音の間（谷）の最小値を雑音床の推定に使う
      const v = Math.round(((k + 0.5) * f0) / df);
      let min = Infinity;
      for (let b = Math.max(1, v - halfWidth); b <= Math.min(nBins - 1, v + halfWidth); b++) {
        min = Math.min(min, power[offset + b]);
      }
      if (Number.isFinite(min)) valleys.push(min);
    }
    const noisePerBin = valleys.length ? median(valleys) : 0;
    const denoised = (k) => Math.max(levels[k] - noisePerBin * binCount[k], noisePerBin, 1e-20);

    let upper = 0;
    for (let k = 2; k <= K; k++) upper += levels[k];

    let slope = null;
    if (K >= 3) {
      let sx = 0, sy = 0, sxx = 0, sxy = 0;
      for (let k = 1; k <= K; k++) {
        const x = Math.log2(k);
        const y = 10 * Math.log10(levels[k]);
        sx += x; sy += y; sxx += x * x; sxy += x * y;
      }
      slope = (K * sxy - sx * sy) / (K * sxx - sx * sx);
    }

    // 声道（フォルマント）の影響を補正した H1*−H2*（Iseli & Alwan, 2004）
    const { formants, fs } = estimateFormants(power, offset, nBins, df);
    const h1h2c =
      10 * Math.log10(denoised(1) / denoised(2)) - (tractGainDb(formants, f0, fs) - tractGainDb(formants, 2 * f0, fs));

    return {
      h1h2: 10 * Math.log10(levels[1] / levels[2]),
      h1h2c,
      hrf: 10 * Math.log10(upper / levels[1]),
      slope,
    };
  }

  // 声区の判定に使う追加の特徴量
  //   h1h4, h2h4 : 第1・第2倍音と第4倍音のレベル差 [dB]
  //   shr        : 分数倍音（(k−½)F0）と倍音のピークパワー比 [dB]
  //   h1a3       : 第1倍音と 2〜3.5 kHz で最も強い倍音の差 [dB]
  //   hiLo       : 2〜5 kHz と 1 kHz 未満の倍音パワー比 [dB]
  //   h1max      : 第1倍音と最も強い倍音の差 [dB]
  //   extent     : 雑音より十分強い倍音が届く上限周波数 [kHz]
  //   hnr        : 4 kHz までの倍音ピークと谷の差の平均 [dB]
  //   alpha      : 1〜5 kHz と 50〜1000 Hz のパワー比 [dB]
  //   centroidF0 : 50〜5000 Hz のスペクトル重心 / F0
  //   cpp        : ケプストラムピークの突出度 [dB]（周期性の明瞭さ）
  function registerFeatures(power, offset, nBins, df, sampleRate, f0, cep) {
    const K = Math.floor(Math.min(5000, df * (nBins - 1) - f0) / f0);
    if (K < 2) return null;
    const dB = (v) => 10 * Math.log10(Math.max(v, 1e-20));
    const maxAround = (c, w) => {
      let m = 0;
      for (let b = Math.max(1, c - w); b <= Math.min(nBins - 1, c + w); b++) m = Math.max(m, power[offset + b]);
      return m;
    };
    const halfWidth = Math.max(1, Math.round((0.15 * f0) / df));
    const P = new Float64Array(K + 1);
    const binCount = new Float64Array(K + 1);
    const peak = new Float64Array(K + 1);
    const valley = new Float64Array(K + 1);
    let harmonicPeaks = 0;
    let subharmonicPeaks = 0;
    for (let k = 1; k <= K; k++) {
      let lo = Math.max(1, Math.ceil(((k - 0.5) * f0) / df));
      let hi = Math.min(nBins - 1, Math.floor(((k + 0.5) * f0) / df));
      if (hi < lo) lo = hi = Math.round((k * f0) / df);
      let sum = 0;
      for (let b = lo; b <= hi; b++) sum += power[offset + b];
      P[k] = sum;
      binCount[k] = hi - lo + 1;
      peak[k] = maxAround(Math.round((k * f0) / df), halfWidth);
      const v = Math.round(((k + 0.5) * f0) / df);
      let min = Infinity;
      for (let b = Math.max(1, v - halfWidth); b <= Math.min(nBins - 1, v + halfWidth); b++) min = Math.min(min, power[offset + b]);
      valley[k] = min;
      harmonicPeaks += maxAround(Math.round((k * f0) / df), 1);
      subharmonicPeaks += maxAround(Math.round(((k - 0.5) * f0) / df), 1);
    }
    const finiteValleys = [];
    for (let k = 1; k <= K; k++) if (Number.isFinite(valley[k])) finiteValleys.push(valley[k]);
    const noise = median(finiteValleys);
    const L = new Float64Array(K + 1);
    let strongest = -Infinity;
    for (let k = 1; k <= K; k++) {
      L[k] = dB(P[k]);
      strongest = Math.max(strongest, L[k]);
    }

    let maxA3 = -Infinity;
    let highEnergy = 0;
    let lowEnergy = 0;
    let extent = 0;
    let hnrSum = 0;
    let hnrCount = 0;
    for (let k = 1; k <= K; k++) {
      const f = k * f0;
      if (f >= 2000 && f <= 3500) maxA3 = Math.max(maxA3, L[k]);
      if (f >= 2000 && f <= 5000) highEnergy += P[k];
      if (f < 1000) lowEnergy += P[k];
      if (L[k] > strongest - 40 && P[k] > 10 * noise * binCount[k]) extent = f;
      if (f <= 4000 && Number.isFinite(valley[k])) {
        hnrSum += dB(peak[k]) - dB(valley[k]);
        hnrCount++;
      }
    }

    let bandHigh = 0;
    let bandLow = 0;
    let centroidNum = 0;
    let centroidDen = 0;
    for (let b = 1; b < nBins; b++) {
      const f = b * df;
      const p = power[offset + b];
      if (f >= 50 && f < 1000) bandLow += p;
      if (f >= 1000 && f <= 5000) bandHigh += p;
      if (f >= 50 && f <= 5000) {
        centroidNum += f * p;
        centroidDen += p;
      }
    }

    // CPP: 対数スペクトルのケプストラムで、F0 範囲のピークが回帰直線からどれだけ突き出ているか
    const N = cep.re.length;
    for (let b = 0; b < N; b++) {
      const src = b <= N / 2 ? b : N - b;
      cep.re[b] = dB(power[offset + Math.min(src, nBins - 1)]);
      cep.im[b] = 0;
    }
    cep.fft(cep.re, cep.im);
    const qLo = Math.floor(sampleRate / 1400);
    const qHi = Math.ceil(sampleRate / 60);
    let cepPeak = -Infinity;
    let cepPeakQ = 0;
    let sx = 0, sy = 0, sxx = 0, sxy = 0, n = 0;
    for (let q = qLo; q <= qHi; q++) {
      const c = 20 * Math.log10(Math.abs(cep.re[q]) / N + 1e-12);
      if (c > cepPeak) {
        cepPeak = c;
        cepPeakQ = q;
      }
      sx += q; sy += c; sxx += q * q; sxy += q * c; n++;
    }
    const trend = (n * sxy - sx * sy) / (n * sxx - sx * sx);

    return {
      h1h4: K >= 4 ? L[1] - L[4] : NaN,
      h2h4: K >= 4 ? L[2] - L[4] : NaN,
      shr: dB(subharmonicPeaks) - dB(harmonicPeaks),
      h1a3: Number.isFinite(maxA3) ? L[1] - maxA3 : NaN,
      hiLo: highEnergy > 0 && lowEnergy > 0 ? dB(highEnergy) - dB(lowEnergy) : NaN,
      h1max: L[1] - strongest,
      extent: extent / 1000,
      hnr: hnrCount ? hnrSum / hnrCount : NaN,
      alpha: dB(bandHigh) - dB(bandLow),
      centroidF0: centroidDen > 0 ? centroidNum / centroidDen / f0 : NaN,
      cpp: cepPeak - (trend * cepPeakQ + (sy - trend * sx) / n),
    };
  }

  // 特徴量 → 裏声である確率
  function falsettoProbability(values) {
    const m = REGISTER_MODEL;
    let x = m.features.map((key, j) => {
      const v = values[key];
      return ((Number.isFinite(v) ? v : m.medians[j]) - m.mean[j]) / m.scale[j];
    });
    m.layers.forEach((layer, li) => {
      const next = layer.b.map((bias, o) => {
        let s = bias;
        for (let j = 0; j < x.length; j++) s += layer.W[j][o] * x[j];
        return s;
      });
      x = li < m.layers.length - 1 ? next.map((v) => Math.max(0, v)) : next;
    });
    return 1 / (1 + Math.exp(-x[0]));
  }

  // 裏声スコア（しきい値からの対数オッズ）。正なら裏声
  function registerScore(values) {
    const logit = (p) => Math.log(p / (1 - p));
    const p = Math.min(1 - 1e-6, Math.max(1e-6, falsettoProbability(values)));
    return logit(p) - logit(REGISTER_MODEL.threshold);
  }

  // 各特徴量が地声寄りか裏声寄りか（UI 表示用）
  function featureLeaning(key, value) {
    const f = REGISTER_MODEL.chips && REGISTER_MODEL.chips[key];
    if (!f || value === null || !Number.isFinite(value)) return null;
    return f.dir * (value - f.threshold) > 0 ? 'falsetto' : 'chest';
  }

  // 4.6〜7.5 kHz と 2.6〜3.4 kHz の 1 ビンあたり平均パワーの比 [dB]。
  // Bluetooth ヘッドセットの通話モード（8 kHz サンプリング）では 4 kHz 付近で音が急に途切れる。
  // 隣り合う高めの帯域どうしを比べるので、母音や声区による高域の強弱には左右されにくい。
  function highBandLevel(power, nBins, df, frames) {
    const bin = (f) => Math.min(nBins - 1, Math.round(f / df));
    if (df * (nBins - 1) < 7500 || frames.length === 0) return NaN;
    let ref = 0;
    let high = 0;
    for (const i of frames) {
      const off = i * nBins;
      for (let b = bin(2600); b <= bin(3400); b++) ref += power[off + b];
      for (let b = bin(4600); b <= bin(7500); b++) high += power[off + b];
    }
    ref /= bin(3400) - bin(2600) + 1;
    high /= bin(7500) - bin(4600) + 1;
    return 10 * Math.log10(high / Math.max(ref, 1e-30) + 1e-20);
  }

  // 連続する有声区間 [start, end) の一覧
  function voicedRuns(voiced) {
    const runs = [];
    let start = -1;
    for (let i = 0; i <= voiced.length; i++) {
      const v = i < voiced.length && voiced[i];
      if (v && start < 0) start = i;
      if (!v && start >= 0) {
        runs.push([start, i]);
        start = -1;
      }
    }
    return runs;
  }

  function medianFilterRuns(values, runs, width) {
    const out = Float32Array.from(values);
    const r = width >> 1;
    for (const [s, e] of runs) {
      for (let i = s; i < e; i++) {
        const win = [];
        for (let j = Math.max(s, i - r); j < Math.min(e, i + r + 1); j++) {
          if (Number.isFinite(values[j])) win.push(values[j]);
        }
        if (win.length) out[i] = median(win);
      }
    }
    return out;
  }

  // ---------------------------------------------------------------- main

  function analyze(input, sampleRate, options) {
    const o = Object.assign({}, DEFAULTS, options);
    const x = removeDC(input, sampleRate);
    const N = o.frameSize;
    const hop = o.hopSize;
    const W = o.pitchFrameSize;
    const nFrames = Math.max(1, Math.floor((x.length - 1) / hop) + 1);
    const nBins = N / 2 + 1;
    const df = sampleRate / N;

    const win = new Float64Array(N);
    let winSum = 0;
    for (let n = 0; n < N; n++) {
      win[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / N);
      winSum += win[n];
    }
    const norm = (winSum * winSum) / 4; // 振幅 1 の正弦波 → 0 dB

    const fft = createFFT(N);
    const re = new Float64Array(N);
    const im = new Float64Array(N);
    const detect = createPitchDetector(sampleRate, o);
    const pitchFrame = new Float64Array(W);

    const power = new Float32Array(nFrames * nBins);
    const f0 = new Float32Array(nFrames);
    const clarity = new Float32Array(nFrames);
    const rmsDb = new Float32Array(nFrames);
    let maxPower = 1e-20;

    for (let i = 0; i < nFrames; i++) {
      const center = i * hop;

      for (let n = 0, idx = center - N / 2; n < N; n++, idx++) {
        re[n] = idx >= 0 && idx < x.length ? x[idx] * win[n] : 0;
        im[n] = 0;
      }
      fft(re, im);
      const off = i * nBins;
      for (let b = 0; b < nBins; b++) {
        const p = (re[b] * re[b] + im[b] * im[b]) / norm;
        power[off + b] = p;
        if (p > maxPower) maxPower = p;
      }

      let energy = 0;
      for (let n = 0, idx = center - W / 2; n < W; n++, idx++) {
        const v = idx >= 0 && idx < x.length ? x[idx] : 0;
        pitchFrame[n] = v;
        energy += v * v;
      }
      rmsDb[i] = 10 * Math.log10(energy / W + 1e-20);
      const p = detect(pitchFrame);
      f0[i] = p.f0;
      clarity[i] = p.clarity;
    }

    // 有声判定
    let loudest = -Infinity;
    for (let i = 0; i < nFrames; i++) loudest = Math.max(loudest, rmsDb[i]);
    const voiced = new Uint8Array(nFrames);
    for (let i = 0; i < nFrames; i++) {
      voiced[i] =
        f0[i] >= o.minF0 &&
        f0[i] <= o.maxF0 &&
        clarity[i] >= o.clarityThreshold &&
        rmsDb[i] >= loudest + o.relativeSilenceDb &&
        rmsDb[i] >= o.absoluteSilenceDb
          ? 1
          : 0;
    }
    for (const [s, e] of voicedRuns(voiced)) {
      if (e - s < o.minVoicedRun) voiced.fill(0, s, e);
    }
    let runs = voicedRuns(voiced);

    // F0 のスパイク除去
    const f0Smooth = medianFilterRuns(f0, runs, 5);
    for (let i = 0; i < nFrames; i++) if (!voiced[i]) f0Smooth[i] = 0;

    // 通話モード（狭帯域）の入力かどうか。判定モデルの入力にも使う
    const voicedFrames = [];
    for (let i = 0; i < nFrames; i++) if (voiced[i]) voicedFrames.push(i);
    const highBandDb = highBandLevel(power, nBins, df, voicedFrames);
    const narrowband = highBandDb < NARROWBAND_DB;
    // モデル入力用（学習時と同じ範囲に丸める。帯域の欠け具合を連続値で伝える）
    const bandInput = Math.max(-60, Math.min(10, highBandDb));

    // フレームごとの特徴量とスコア
    const h1h2 = new Float32Array(nFrames).fill(NaN);
    const h1h2c = new Float32Array(nFrames).fill(NaN);
    const hrf = new Float32Array(nFrames).fill(NaN);
    const slope = new Float32Array(nFrames).fill(NaN);
    const rawScores = new Float32Array(nFrames).fill(NaN);
    const extras = {};
    for (const key of EXTRA_FEATURES) extras[key] = new Float32Array(nFrames).fill(NaN);
    const cep = { fft, re: new Float64Array(N), im: new Float64Array(N) };
    for (let i = 0; i < nFrames; i++) {
      if (!voiced[i]) continue;
      const feat = harmonicFeatures(power, i * nBins, nBins, df, f0Smooth[i], o.harmonicMaxFreq);
      if (!feat) {
        voiced[i] = 0;
        f0Smooth[i] = 0;
        continue;
      }
      h1h2[i] = feat.h1h2;
      h1h2c[i] = feat.h1h2c;
      hrf[i] = feat.hrf;
      slope[i] = feat.slope === null ? NaN : feat.slope;
      const extra = registerFeatures(power, i * nBins, nBins, df, sampleRate, f0Smooth[i], cep);
      if (extra) for (const key of EXTRA_FEATURES) extras[key][i] = extra[key];
      rawScores[i] = registerScore({ ...feat, ...(extra || {}), highBandDb: bandInput });
    }
    runs = voicedRuns(voiced);
    const scores = medianFilterRuns(rawScores, runs, o.scoreSmoothing);

    // 0: 無声, 1: 地声, 2: 裏声
    const labels = new Uint8Array(nFrames);
    const voicedIdx = [];
    let falsettoCount = 0;
    for (let i = 0; i < nFrames; i++) {
      if (!voiced[i]) continue;
      voicedIdx.push(i);
      labels[i] = scores[i] > 0 ? 2 : 1;
      if (labels[i] === 2) falsettoCount++;
    }

    const pick = (arr) => median(voicedIdx.map((i) => arr[i]).filter(Number.isFinite));
    const voicedSeconds = (voicedIdx.length * hop) / sampleRate;
    let summary;
    if (voicedSeconds < 0.2) {
      summary = { label: null, voicedSeconds };
    } else {
      const ratio = falsettoCount / voicedIdx.length;
      const meanScore = voicedIdx.reduce((s, i) => s + scores[i], 0) / voicedIdx.length;
      const features = { h1h2: pick(h1h2), h1h2c: pick(h1h2c), hrf: pick(hrf), slope: pick(slope) };
      for (const key of EXTRA_FEATURES) features[key] = pick(extras[key]);
      const medianF0 = pick(f0Smooth);
      summary = {
        label: ratio > 0.5 || (ratio === 0.5 && meanScore > 0) ? 'falsetto' : 'chest',
        falsettoRatio: ratio,
        meanScore,
        voicedSeconds,
        medianF0,
        note: noteName(medianF0),
        highBandDb,
        narrowband,
        features,
        leaning: Object.fromEntries(Object.keys(REGISTER_MODEL.chips || {}).map((key) => [key, featureLeaning(key, features[key])])),
      };
    }

    return {
      sampleRate,
      duration: x.length / sampleRate,
      frameSize: N,
      hopSize: hop,
      nFrames,
      nBins,
      binHz: df,
      highBandDb,
      power,
      maxPower,
      f0: f0Smooth,
      voiced,
      scores,
      labels,
      features: { h1h2, h1h2c, hrf, slope, ...extras },
      summary,
    };
  }

  const api = { analyze, noteName, createFFT, estimateFormants, falsettoProbability, DEFAULTS, REGISTER_MODEL };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.VoiceAnalyzer = api;
})(typeof self !== 'undefined' ? self : this);
