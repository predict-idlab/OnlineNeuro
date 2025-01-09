function result = axonsim_call(varin)
    % Varin should be a struct with the stimulation parameters:

    % Electrode position: e_pos [mm] as array [1+, 1-, 2+, 2-, 3+, 3- ...]
    % Fiber curvature: c [-1 to 1] usually set to 0
    % Medium resistance: phi [ohm/cm] usually 35
    % Fiber diameter: dia [um]
    % Time of simulation: sim_dur [ms]
    % Pulse duration: pulse_dur [ms]
    % Stimulation frequency: frq [Hz]
    % Electrode separation: e_sep [mm]
    % Electrode type: e_type [mm]
    % Simulation accuracy: accuract [mA]
    % Fiber length: length [uA]
    % Calculate threshold: calc_thr [int] (should be set to 0)
    % Stimulation function type: fun_type [string]: single pulse, double pulse or custom
    % Model type: model_type [str]: MRG, CRRSS, McNeal:
    % Custom function name: custom_fun [str] specified as "custom_fun(t, varin)"
        % should have all arguments specified. For example for stim_sin function
        % it should be specified as "stim_sin(t, 20000, 1)"
    % Current amplitude: I [mA]

    % Get stimulation parameters
    % For each field in varin, if a value was not passed, then use a
    % default.
    % This function RETURNS the final struct to be able to map default
    % values as well (may be usefull for batch simulations and offline
    % training).

    %epos [0, 1]?
    %sim_dur probably needs to be larger (default)
    %e_type suggest a mm? - it is actually a switch 1: single, 2 double, 3
    %triple, 4 intracellular,
    % fiber length [uA?] it should be um\
    % calc_thr purpose?

    
    % This is not handled when loading the defaults via the .json

    % default_struct = struct;
    % default_struct.e_pos = [-4, 1, 4, 1];
    % default_struct.c = 0;
    % default_struct.phi = 35;
    % default_struct.dia = 10;
    % default_struct.sim_dur = 5;
    % default_struct.pulse_dur = [0.1, 0.1];
    % default_struct.frq = 1;
    % default_struct.e_sep = [0.5, 0.5];
    % default_struct.e_type = ["single", "single"]; %'single', 'double, 'triple', 'intracellular'
    % default_struct.accuracy = 0.01;
    % default_struct.length = 20000;
    % default_struct.calc_thr = 0;
    % default_struct.fun_type = ["custom", "single pulse"]; %'single_pulse', 'double pulse', 'custom'
    % default_struct.model_type = "MRG";
    % default_struct.custom_fun = ["stim_sin(t, 20000, 1)", "stim_sin(t, 2000, 1)"];
    % default_struct.I = [-1, -1];
    % fib_length [um]
    
    e_pos = varin.e_pos;
    c = varin.c;
    phi = varin.phi;
    dia = varin.dia;
    sim_dur = varin.sim_dur;
    pulse_dur = varin.pulse_dur;
    frq = varin.frq;
    e_sep = varin.e_sep;
    e_type = varin.e_type;
    accuracy = varin.accuracy;
    fib_length = varin.length;
    calc_thr = varin.calc_thr;
    fun_type = varin.fun_type;
    model_type = varin.model_type;
    custom_fun = varin.custom_fun;

    % x = 2;
    % dummy_fun = @(t) sin(t) + x;
    % custom_fun{1} = "dummy_fun(t)";
    % fun_type{1} = "custom";
    I = varin.I;

    
    %params = varin.custom_fun_params;
    
    %
    n_sources = varin.num_electrodes;
    e_sep = e_sep*1000;
    switch string(model_type)
        case "MRG"
            model_nr = 1;
        case "CRRSS"
            model_nr = 2;
        case "McNeal"
            model_nr = 3;
    end
    
    %Modified code from axonsim_mod
    % Handling single strings as arrays
    fun_type_array = NaN(1, n_sources);
    e_type_array = NaN(1, n_sources);

    for i=1:n_sources
        switch string(e_type(i))
            case "single"
                e_type_array(i) = 1;
            case "double"
                e_type_array(i) = 2;
            case "triple"
                e_type_array(i) = 3;
            case "intracellular"
                e_type_array(i) = 4;
        end
        
        switch string(fun_type(i))
            case {"single_pulse", "single pulse"}
                fun_type_array(i) = 1;
            case {"double_pulse", "double pulse"}
                fun_type_array(i) = 2;
            case "custom"
                fun_type_array(i) = 3;
            otherwise
                fun_type_array(i) = 3;
        end
    end
    %e_type = str2double(e_type);
    %fun_type = str2double(fun_type);
    e_type = e_type_array;
    fun_type = fun_type_array;
    e_pos = e_pos * 1000;  %um
    phi = phi*100000; % ohm um
    
    for i=1:n_sources
        if e_type(i) ~= 4
            I(i) = I(i) /1000; %A
            accuracy = accuracy/1000;
        end
    end

    [x,y] = curvature(c,fib_length);
    l=0;

    s = size(x);
    for i = 2:s(2)
       l = [l,l(i-1)+sqrt((x(i)-x(i-1))^2+(y(i)-y(i-1))^2)];
    end

    x_m = x'/1000000;
    y_m = y'/1000000;
    z_m = zeros(s(2),s(1));

    V=[];
    sI = [];
    for i = 1:n_sources
        switch e_type(i)
            case 1
                V(i,:)=electrode(phi,I(i),e_pos(i*2 -1),e_pos(i*2),x,y);
            case 2
                V(i,:)=electrode(phi,I(i),e_pos(i*2 -1)-e_sep(i)/2,e_pos(i*2),x,y)-electrode(phi,I,e_pos(i*2 -1)+e_sep(i)/2,e_pos(i*2),x,y);
            case 3
                V(i,:)=electrode(phi,I(i),e_pos(i*2 -1),e_pos(i*2),x,y)-0.5*electrode(phi,I,e_pos(i*2 -1)-e_sep(i),e_pos(i*2),x,y)-0.5*electrode(phi,I,e_pos(i*2 -1)+e_sep(i),e_pos(i*2),x,y);
            case 4
                V(i,:)=zeros(1,s(2));
        end
        sI(i) = sign(I(i));
        I(i) = abs(I(i));
    end

    data = [x_m,y_m,z_m,V'];

    if e_type(i) == 4
        [t,Y,N] = model_mod(model_nr,sim_dur,data,pulse_dur,fun_type,custom_fun,dia,frq,0,sI*I(i),e_pos(i*2 -1:i*2)/1000+8, n_sources);
    else
        [t,Y,N] = model_mod(model_nr,sim_dur,data,pulse_dur,fun_type,custom_fun,dia,frq,0, n_sources);
    end


    V_tot = sum(V,1);
    af = [];
    for i = 2:s(2)-1
        af(i-1) = (V_tot(i)-V_tot(i-1))/(l(i)-l(i-1))^2+(V_tot(i)-V_tot(i+1))/(l(i)-l(i+1))^2;
    end

    szy = size(Y);

    m = 16;
    
    Yp=zeros(szy(1),N-m);

    for i = 1:N-m
        Yp(:,i) = Y(:,i+m/2);%-(i-1)*40; No scaling done (this can be handled externally)
    end

    isi = 1/frq*1000;
    stimuli = [];
    for j = 1:n_sources
        ts=[];
        stimf=[];
        for i = 0:1/1000:sim_dur
            stimf=[stimf,sI(j)*stimulation_fun(mod(i,isi),pulse_dur(j),fun_type(j),custom_fun(j))];
            ts=[ts,i];
        end
        stimuli(j,:) = stimf;
    end

    AP = Y(:,floor(N/2));
    % fname = sprintf('AP_%s.mat', datestr(now,'mm-dd-yyyy HH-MM'));
    % save(fname, "AP");
    %
    st = stimuli(:,1:10:end);
    % st = st(1:end-1);
    % fname = sprintf('stimf_%s.mat', datestr(now,'mm-dd-yyyy HH-MM'));
    % save(fname, "st");

    tosave = struct;
    tosave.varin = varin;
    tosave.N = N;
    tosave.Yp = Yp;
    tosave.V_tot = V_tot;
    tosave.af = af;
    tosave.stimuli = stimuli;
    tosave.ts = ts;

    
    currentFolder = pwd;
    relativeFolder = fullfile(currentFolder, '../../simulations/axonsim_nerve_block/full_mats/');

    fname = sprintf('%s/simulation_%s.mat', relativeFolder, datestr(now,'mm-dd-yyyy HH-MM'));
    %display(fname)
    tosave.fname = fname;
    save(fname,'tosave');

    result = tosave;

end
