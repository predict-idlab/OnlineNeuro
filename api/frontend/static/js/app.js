const globalData = {
    model: null,
    experiment: null,
    optimizer:null
}

let optimizableExperimentFeatures
let userDefinedExperimentFeatures
let defaultFeaturesExperimentFeatures

const defaultFeatures = {
    model: null,
    optimizer: null
}

function toggleSidebar(selector) {
    document.querySelector(selector).classList.toggle('sidebar-closed');
}

function showSpinner(show) {
    let spinner = document.getElementById('spinner');

    spinner.style.display = show ? 'block' : 'none';
}

function capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

function cleanFinalParameters(params) {
    const cleanedParams = {};

    Object.entries(params).forEach(([key, subStruct]) => {
        if (key.startsWith("toggle_")) {
            return;
        }
        const hasMinMax = 'min_value' in subStruct && 'max_value' in subStruct;
        const values = subStruct.value || false;
        const minValues = subStruct.min_value || false;
        const maxValues = subStruct.max_value || false;

        if (!hasMinMax) {
            // Scenario 1: Only 'value' key exists
            const nonNullValues = Array.isArray(values)
                ? values.filter(val => val !== null)
                : (values !== null ? [values] : []);
            if (nonNullValues.length > 0){
                cleanedParams[key] = nonNullValues
            }
        } else {
            // Scenario 2: 'value', 'min_value', and 'max_value' keys exist
            const valuesIsArray = Array.isArray(values);
            const minValuesIsArray = Array.isArray(minValues);
            const maxValuesIsArray = Array.isArray(maxValues);

            const anyArray = valuesIsArray || minValuesIsArray || maxValuesIsArray
            const maxLength =  Math.max(valuesIsArray ? values.length : 1,
                minValuesIsArray ? minValues.length : 1,
                maxValuesIsArray ? maxValues.length : 1)

            const filledValues = new Array(maxLength).fill(0);
            const filledMinValues = new Array(maxLength).fill(0);
            const filledMaxValues = new Array(maxLength).fill(0);


            for (let i = 0; i < maxLength; i++) {
                values && (filledValues[i] = values);
                minValues && (filledMinValues[i] = minValues);
                maxValues && (filledMaxValues[i] = maxValues);
            }

            cleanedParams[key] = {
                value: filledValues,
                min_value: filledMinValues,
                max_value: filledMaxValues,
            };
        }
    });

    return cleanedParams;
}

function restructureParameters(data) {
    const finalParameters = {};

    Object.entries(data).forEach(([key, value]) => {
        if (typeof value === "string" && !isNaN(value)) {
            value = parseFloat(value);
        }

        if (key.startsWith('pulse')){
            if (!finalParameters.pulses) {
                finalParameters.pulses = [];
            }
            // Directly copy the pulse parameters into the pulses array
            finalParameters.pulses.push(value);
        } else {
            const cleanedKey = key
            .replace(/_\d+$/, "") // Remove the index (_n)
            .replace(/_(min|max)$/, "")    // Remove "_min" or "_max" if present
            .replace(/_[a-zA-Z]$/, "");    // Remove the last "_x" (it would be a dimension)

            const typeMatch = key.match(/_(min|max)/); // Check if key ends with "_min" or "_max"
            const type = typeMatch ? `${typeMatch[1]}_value` : "value"; // Set the key type
            const ixMatch = key.match(/_(\d+)$/);
            const ix = ixMatch ? parseInt(ixMatch[1], 10) : null; // Extract the index as an integer, or null if none

            if (!finalParameters[cleanedKey]) {
                finalParameters[cleanedKey] = {};
            }
            if (!finalParameters[cleanedKey][type]) {
                finalParameters[cleanedKey][type] = [];
            }

            let dimensions
            if (cleanedKey in globalData["experiment"]) {
                dimensions = globalData["experiment"][cleanedKey].dimensions || []
            } else {
                dimensions = []
            }

            if (dimensions.length === 0) {
                if (ix !== null) {
                    finalParameters[cleanedKey][type][ix] = value;
                } else {
                    finalParameters[cleanedKey][type] = value;
                }
            } else{
                const dimensionMatch = key.match(/_([a-zA-Z])_\d+$/); // Matches the dimension (_x_, _y_, etc.)
                const dimension = dimensionMatch[1];
                const dimensionIndex = dimensions.indexOf(dimension);

                if (!finalParameters[cleanedKey][type][dimensionIndex]){
                    finalParameters[cleanedKey][type][dimensionIndex] = []
                }
                finalParameters[cleanedKey][type][dimensionIndex][ix] = value;
            }
        }
    });

    const cleanParameters = cleanFinalParameters(finalParameters)
    console.log("Clean Params", cleanParameters)

    Object.entries(defaultFeaturesExperimentFeatures).forEach(([key, value]) => {
        if (!(key in cleanParameters)) {
            cleanParameters[key] = value;
        }
    });

    return cleanParameters;
}

function collectInputs(elementId, prefix=null) {
    const sidebarFrame = document.getElementById(elementId);
    // Object to store the final parameters
    const parameters = {};
    parameters.otherParameters = {};

    // === 1 Collect all non-prefixed parameters ===
    const inputs = sidebarFrame.querySelectorAll('input');
    inputs.forEach(input => {
        if (!prefix || !input.id.startsWith(prefix)) {
            parameters.otherParameters[input.id] = input.type === 'checkbox' ? input.checked : input.value;
        }
    });

    const selects = sidebarFrame.querySelectorAll('select');
    selects.forEach(select => {
        if (!prefix || !select.id.startsWith(prefix)) {
            parameters.otherParameters[select.id] = select.options[select.selectedIndex].value;
        }
    });

     // === 2 Collect prefixed, grouped parameters (like pulse_0_x, pulse_1_y...) ===
    if (prefix) {
        const prefixedKeys = Object.keys(globalData["experiment"]).filter(key => key.startsWith(prefix));
        const grouped = {};

        prefixedKeys.forEach(fullKey => {
            const parts = fullKey.split('_');
            const groupIndex = parts[1]; // e.g. 0, 1, 2
            const paramKey = parts.slice(2).join('_'); // e.g. I, freq, etc.

            if (!grouped[groupIndex]) grouped[groupIndex] = {};
            grouped[groupIndex][paramKey] = globalData["experiment"][fullKey];
        });

        const groupedList = [];

        Object.keys(grouped).forEach(groupIndex => {
            const config = grouped[groupIndex];
            const groupData = {};

            Object.keys(config).forEach(paramKey => {
                const baseId = `${prefix}${groupIndex}_${paramKey}`;
                const inputEl = sidebarFrame.querySelector(`#${baseId}`);
                const minEl = sidebarFrame.querySelector(`#${baseId}_min`);
                const maxEl = sidebarFrame.querySelector(`#${baseId}_max`);

                // Initialize entry
                groupData[paramKey] = {};

                if (inputEl) {
                    if (inputEl.type === 'checkbox') {
                        groupData[paramKey].value = inputEl.checked;
                    } else if (inputEl.tagName === 'SELECT') {
                        groupData[paramKey].value = inputEl.options[inputEl.selectedIndex].value;
                    } else {
                        groupData[paramKey].value = inputEl.value;
                    }
                }

                if (minEl) groupData[paramKey].min_value = minEl.value;
                if (maxEl) groupData[paramKey].max_value = maxEl.value;
            });

            groupedList.push(groupData);
        });

        if (groupedList.length > 0) {
            parameters[`${prefix.replace(/_$/, '')}Parameters`] = groupedList;
        }
    }

    // === 3 Combine and return ===
    return parameters;
}

async function startExperiment() {
    showSpinner(true);
    // Collect all inputs from the "sidebar-frame" div
    let parameters = {};
    let leftbarParameters = collectInputs('sidebar-left', 'pulse_')
    parameters.experimentParameters = restructureParameters(leftbarParameters.otherParameters)
    parameters.experimentParameters['experiment'] = document.getElementById('experiment').value || null
    parameters.pulseParameters = leftbarParameters.pulseParameters || []

    let rightbarParametersModel = collectInputs('parameters-model')
    parameters.modelParameters = restructureParameters(rightbarParametersModel.otherParameters)
    let rightbarParametersOptimizer = collectInputs('parameters-model')
    parameters.optimizerParameters = restructureParameters(rightbarParametersOptimizer.otherParameters)

    await createIframes()

    try {
        const response = await fetch('/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(parameters),
        });
        const data = await response.json();
        console.log("Response:", data);

    } catch(error) {
        console.error('Error starting the experiment:', error);
    } finally {
            showSpinner(false);
    }
}

async function checkStatus() {
    const response = await fetch('/status');
    const data = await response.json();
    console.log(data);
}

async function stopExperiment() {
    showSpinner(true);
    const response = await fetch('/stop', { method: 'POST' });
    const data = await response.json();
    console.log(data);
    showSpinner(false);
}

async function fetchData(endpoint, selectId, defaultText) {
    try {
        const response = await fetch(`/${endpoint}`); // API route that returns JSON
        const items = await response.json(); // Ensure the response is parsed as JSON
        const selectElement = document.getElementById(selectId);

        // Clear existing options
        selectElement.innerHTML = `<option value="">${defaultText}</option>`; // Reset to default option

        // Populate the select element with the fetched items
        items.forEach(item => {
            const option = document.createElement('option');
            option.value = item; // Set the value for the option
            option.innerText = item; // Display the item name
            selectElement.appendChild(option); // Append the option to the select
        });
    } catch (error) {
        console.error(`Error fetching ${endpoint}:`, error);
    }
}
function sortFeatures(features, dataDic) {
    return features.sort((a, b) => {
        if (dataDic[a].type < dataDic[b].type) return -1;
        if (dataDic[a].type > dataDic[b].type) return 1;
        if (a < b) return -1;
        if (a > b) return 1;

        return 0; // if they are equal
    });
}
async function updatePulseParameters(){
    let pulseTypes = Array.from(document.querySelectorAll('[id^="fun_type_"]')).reduce((acc, input) => {
        const match = input.id.match(/_(\d+)$/); // Extract the number at the end of the ID
        if (match) {
            const number = match[1]; // Get the matched number
            acc[number] = input.value; // Assign the input value to the corresponding number
        }
        return acc; // Return the accumulator for the next iteration
    }, {});

    if (Object.keys(pulseTypes).length === 0){
        return;
    }
    let configs = [];
    await Promise.all(Object.values(pulseTypes).map(async pt => { // Use Object.values to iterate
        const response = await fetch('/get_fun_parameters', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({function: pt}) // Pass the experiment as JSON in the body
        });
        if (!response.ok) {
            console.error("Error fetching parameters:", response.statusText);
            return;
        }
        let this_config = await response.json()
        configs.push(this_config)
        }));

    // Clear out any previous pulse keys in globalData to prevent leftover values
    Object.keys(globalData["experiment"]).forEach(key => {
        if (key.startsWith('pulse_')) {
            delete globalData["experiment"][key];
        }
    });

    for (let [i, pulseConfig] of configs.entries()) {
        let pulse_key = `pulse_${i}`
        const prefixedPulseConfig = {};

        Object.keys(pulseConfig).forEach(key => {
            const prefixedKey = `${pulse_key}_${key}`; // Create the prefixed key
            globalData["experiment"][prefixedKey] = pulseConfig[key]; // Store in globalData['experiment']
            prefixedPulseConfig[prefixedKey] = pulseConfig[key]; // Map with prefixed keys

        });

        let optPulseFeatures = Object.keys(prefixedPulseConfig).filter(
            key => prefixedPulseConfig[key]?.optimizable === true
        );
        let fixPulseFeatures = Object.keys(prefixedPulseConfig).filter(
            key => prefixedPulseConfig[key]?.user_fixed === true
        );

        optPulseFeatures = sortFeatures(optPulseFeatures, prefixedPulseConfig);
        fixPulseFeatures = sortFeatures(fixPulseFeatures, prefixedPulseConfig);

        await updateFixedParameters(fixPulseFeatures, prefixedPulseConfig, 'parameters-container-fixed-pulse', i)
        await updateVariableParameters(optPulseFeatures, prefixedPulseConfig, 'parameters-container-variable-pulse', i)
    }

    Object.keys(pulseTypes).forEach(number => {
        const input = document.getElementById(`fun_type_${number}`);
        input.addEventListener('change', cleanParametersAndUpdate);  // Call the function on change
    });

}
async function cleanParametersAndUpdate(){
    // TODO here
    const containerFixedPulse = document.getElementById('parameters-container-fixed-pulse');
    const containerVariablePulse = document.getElementById('parameters-container-variable-pulse');
    containerFixedPulse.innerHTML = ''; //Clear existing parameters
    containerVariablePulse.innerHTML = ''; //Clear existing parameters
    await updatePulseParameters();
}
async function updateParameters(){
    await updateFixedParameters(userDefinedExperimentFeatures, globalData['experiment'], 'parameters-container-fixed')
    await updateVariableParameters(optimizableExperimentFeatures, globalData['experiment'] , 'parameters-container-rest')
    await updatePulseParameters()
}

async function updateParametersId(id_name){
    const data = globalData[id_name];
    if (!data) {
        console.warn(`No data available for ${id_name}`);
        return;
    }
    const keyList = Object.keys(data).filter(k => data[k].user_fixed === true);
    const containerName = `parameters-${id_name}`;

    await updateFixedParameters(keyList, data, containerName)
}

// Listeners for the plotting of the electrodes (disabled atm)
async function updateElectrodeListeners(){
    const e_pos_inputs = document.querySelectorAll('[id^="e_pos"]');
    // e_pos_inputs.forEach(input => {
    //     input.addEventListener('change', drawPlot);
    // });
    const diameterInput = document.getElementById('dia') || document.getElementById('axon_diameter');
    //TODO uncomment this once the other thing is fixe
    // diameterInput.addEventListener('change', drawPlot);
    // await drawPlot()
}

async function addPlotListeners(){
    const experimentSelect = document.getElementById('experiment');
    experimentSelect.addEventListener('change', addPlotListeners);

    const num_electrodes = document.getElementById('num_electrodes');
    if (num_electrodes){
        num_electrodes.addEventListener('change', updateElectrodeListeners);
        await updateElectrodeListeners()
    }

}
async function experimentChange(){
    await loadExperimentParameters()
    if (globalData["experiment"].hasOwnProperty('num_electrodes')) {
        await addPlotListeners();
        // TODO uncomment
        //drawPlot();
    } else {
        // Clear the canvas if no electrodes are present
        const canvas = document.getElementById('plotCanvas');
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear the canvas
        console.log("No electrodes to plot.");
    }
}

async function modelChange(){
    await loadParameters("model", {selectable:true});
}

async function loadExperimentParameters() {
    const selectedExperiment = document.getElementById('experiment').value;

    if (selectedExperiment===""){
        globalData["experiment"] = false
        const container = document.getElementById('parameters-container-num');
        const fixedTitle = document.getElementById('parameters-header-fixed');
        const variableTitle = document.getElementById('parameters-header-rest');
        const containerFixed = document.getElementById('parameters-container-fixed');
        const containerVariable = document.getElementById('parameters-container-rest');
        const containerFixedPulse = document.getElementById('parameters-container-fixed-pulse');
        const containerVariablePulse = document.getElementById('parameters-container-variable-pulse');

        container.innerHTML = '';  // Clear existing parameters
        fixedTitle.innerHTML = '';  // Clear existing parameters
        variableTitle.innerHTML = '';  // Clear existing parameters
        containerFixed.innerHTML = '';  // Clear existing parameters
        containerVariable.innerHTML = '';  // Clear existing parameters
        containerFixedPulse.innerHTML = ''; //Clear existing parameters
        containerVariablePulse.innerHTML = ''; //Clear existing parameters

    } else{
        const response = await fetch('/get_parameters', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                type: "problem_parameters",
                experiment: selectedExperiment}) // Pass the experiment as JSON in the body
        });

        if (!response.ok) {
            console.error("Error fetching parameters:", response.statusText);
            return;
        }
        globalData["experiment"] = await response.json();

        console.log("Received load")
        console.log(globalData["experiment"])
        const excludeList = ['num_electrodes'];

        optimizableExperimentFeatures = Object.keys(globalData["experiment"]).filter(key => globalData["experiment"][key].optimizable ===true)
        userDefinedExperimentFeatures = Object.keys(globalData["experiment"]).filter(key => globalData["experiment"][key].user_fixed ===true && !excludeList.includes(key))
        // Non GUI values (see the experiment .json for optimizable/user_fixed both false.)
        defaultFeaturesExperimentFeatures = Object.fromEntries(
            Object.entries(globalData["experiment"])
                .filter(([key, value]) => value.user_fixed === false && value.optimizable === false)
                .map(([key, value]) => [key, value.value]) // Keep only the 'value' key
        );

        optimizableExperimentFeatures = sortFeatures(optimizableExperimentFeatures, globalData["experiment"]);
        userDefinedExperimentFeatures = sortFeatures(userDefinedExperimentFeatures, globalData["experiment"]);

        const container = document.getElementById('parameters-container-num');
        const containerFixed = document.getElementById('parameters-container-fixed');
        const containerVariable = document.getElementById('parameters-container-rest');
        const containerFixedPulse = document.getElementById('parameters-container-fixed-pulse');
        const containerVariablePulse = document.getElementById('parameters-container-variable-pulse');

        const fixedTitle = document.getElementById('parameters-header-fixed');
        const variableTitle = document.getElementById('parameters-header-rest');

        fixedTitle.innerText = '';
        const hrFixed = document.createElement('hr');
        hrFixed.classList.add('horizontal-line');
        fixedTitle.append(hrFixed)

        const titleFix = document.createElement('div');
        titleFix.classList.add('block-title');
        titleFix.innerText = "Fixed parameters"
        fixedTitle.append(titleFix)

        variableTitle.innerText = '';
        const hrVar = document.createElement('hr');
        hrVar.classList.add('horizontal-line');
        variableTitle.append(hrVar)

        const titleVar = document.createElement('div');
        titleVar.classList.add('block-title');
        titleVar.innerText = "Search space"
        variableTitle.append(titleVar)

        if (container) container.innerHTML = '';  // Clear existing parameters
        if (containerFixed) containerFixed.innerHTML = '';
        if (containerVariable) containerVariable.innerHTML = '';
        if (containerFixedPulse) containerFixedPulse.innerHTML = '';
        if (containerVariablePulse) containerVariablePulse.innerHTML = '';

        if (globalData["experiment"].multiple_sets?.value) { //Extend this so not only num_electrodes?
            // TODO This would be a hard requirements and needs to be documented.
            const row = document.createElement('div');
            row.classList.add('parameter-container')
            row.id = 'electrode-sets-row'

            const label = document.createElement('div');
            label.classList.add('parameter-label');
            label.innerText = "Num. of electrodes"

            const input = document.createElement('input');
            input.type = 'number';
            input.id = 'num_electrodes';
            input.value = globalData["experiment"].num_electrodes.value;
            input.min = globalData["experiment"].num_electrodes.min_value;
            input.max = globalData["experiment"].num_electrodes.max_value;
            input.onchange = updateParameters;

            row.appendChild(label)
            row.appendChild(input)

            container.append(row)
        }
        await updateParameters()

    }
}

async function loadParameters(id_name, { selectable = true } = {}) {
    let selectedValue = null;
    const container = document.getElementById(`parameters-${id_name}`);
    container.innerHTML = '';  // Clear existing parameters
    //const selectedModel = document.getElementById('model').value;

    if (selectable){
        selectedValue = document.getElementById(id_name).value;
        if (selectedValue === ""){
            window[`global${capitalize(id_name)}Data`] = false;
            return;
        }
    }

    const bodyPayload = {
        type: id_name,
    };

    if (selectable) {
        bodyPayload[id_name] = selectedValue;

    }
    console.log(bodyPayload)
    const response = await fetch('/get_parameters', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},

        body: JSON.stringify(bodyPayload)
    });

    if (!response.ok) {
        console.error("Error fetching parameters:", response.statusText);
        return;
    }

    const parameterData = await response.json();
    globalData[id_name] = parameterData;

    console.log("Received load")
    console.log(parameterData)

    const userDefined = Object.keys(parameterData)
        .filter(key => parameterData[key].user_fixed === true);

    // Non GUI values (see the model .json for non-user_fixed values.)
    const defaults = Object.fromEntries(
        Object.entries(parameterData)
            .filter(([_, value]) => value.user_fixed === false)
            .map(([key, value]) => [key, value.value])
    );

    defaultFeatures[id_name] = defaults;

    console.log("Default features")
    console.log(`Default ${id_name} features`, defaults);

    // const fixedTitle = document.getElementById('parameters-header-fixed');
    // const variableTitle = document.getElementById('parameters-header-rest');
    await updateParametersId(id_name)

}

function createParameterContainer() {
    const container = document.createElement('div');
    container.classList.add('parameter-container');
    return container;
}

function createLabel(key, type, data) {
    const label = document.createElement('div');
    label.classList.add('parameter-label');
    if (type === 'boolean' || type === 'cat' || type === 'cat-mult' || type === 'complex') {
        label.innerText = `${key}`;
    } else {
        label.innerText = `${key} [${data.min_value}, ${data.max_value}]`;
    }
    return label;
}

function createSelectInput(keyOrId, options, index = null, isMultiple = true) {
    const select = document.createElement('select');
    select.multiple = isMultiple;
    // Determine ID based on input type
    if (index !== null) {
        const key = keyOrId;
        select.id = `${key}_${index}`;
    } else {
        select.id = keyOrId; // Direct ID for simpler cases
    }

    // Populate options
    options.forEach((optionText, i) => {
        const option = document.createElement('option');
        option.value = optionText;
        option.innerText = optionText;
        if (i === 0) option.selected = true; // Select first by default
        select.appendChild(option);
    });

    return select;
}
function createNumericInputRow(key, data, index=null) {
    const inputRow = createInputRow(key, false);
    const valueInput = document.createElement('input');
    valueInput.type = 'number';
    valueInput.id = index !== null ? `${key}_${index}` : `${key}`;
    if (data.value !== null) valueInput.value = data.value;
    if (data.min !== null) valueInput.min = data.min;
    if (data.max !== null) valueInput.max = data.max;

    inputRow.appendChild(valueInput);
    return inputRow;
}

function addFixedNumericMultipleInputs(container, key, data, columns, index = null) {
    const loopStart = index !== null ? index : 0;
    const loopEnd = index !== null ? index + 1 : columns;

    for (let i = loopStart; i < loopEnd; i++) {
        container.appendChild(createNumericInputRow(key, data, i));
    }
}
function addSingleNumericInput(container, key, data, index=null) {
    container.appendChild(createNumericInputRow(key, data, index));
}

function addCatMultInputs(container, key, data, columns, index = null) {
    const loopStart = index !== null ? index : 0;
    const loopEnd = index !== null ? index + 1 : columns;

    for (let i = loopStart; i < loopEnd; i++) {
        const inputRow = createSelectRow(key, data.possible_values, i );
        container.appendChild(inputRow);
    }
}

function addSingleCatInput(container, key, data, index=null) {
    container.appendChild(createSelectRow(key, data.possible_values, index));
}

function addArrayInputs(container, key, data, columns, index=null) {
    for (let i = 0; i < columns; i++) {
        const inputRow = document.createElement('div');
        inputRow.classList.add('input-row');

        const rowLabel = createRowLabel(i);
        inputRow.appendChild(rowLabel);

        data.dimensions.forEach((dim, dimIndex) => {
            const dimLabel = document.createElement('div');
            dimLabel.classList.add('dim-label');
            dimLabel.innerText = `${dim}:`;

            const valueInput = document.createElement('input');
            valueInput.type = 'number';
            valueInput.id = `${key}_${dim}_${i}`;
            valueInput.value = data.value?.[dimIndex] || 0;
            if (data.min_value && data.max_value) {
                valueInput.min = data.min_value[dimIndex];
                valueInput.max = data.max_value[dimIndex];
            }
            inputRow.appendChild(dimLabel);
            inputRow.appendChild(valueInput);
        });

        container.appendChild(inputRow);
    }
}

function createInputRow(key, insertBoolean = false, index=null) {
    const inputRow = document.createElement('div');
    inputRow.classList.add('input-row');

    // Create label for the row, adjusted for index and column count
    const rowLabel = document.createElement('div');
    rowLabel.classList.add('row-ix');

    if (insertBoolean) {
        // Create a checkbox for boolean inputs
        //rowLabel.innerText = "set (True)"
        const booleanInput = document.createElement('input');
        booleanInput.type = 'checkbox';
        booleanInput.id = index=== null? `${key}` : `${key}_${index}`;
        inputRow.appendChild(rowLabel); // Add the label first
        inputRow.appendChild(booleanInput); // Add the checkbox input
    }

    inputRow.appendChild(rowLabel);

    return inputRow;
}

function createSelectRow(key, options, index=null) {
    const inputRow = document.createElement('div');
    inputRow.classList.add('input-row');

    const rowLabel = createRowLabel(index);
    const selectInput = document.createElement('select');
    selectInput.id = index !== null ? `${key}_${index}` : `${key}`;

    options.forEach((value) => {
        const option = document.createElement('option');
        option.value = value;
        option.innerText = value;
        selectInput.appendChild(option);
    });
    inputRow.appendChild(rowLabel);
    inputRow.appendChild(selectInput);
    return inputRow;
}

function createRowLabel(index) {
    const label = document.createElement('div');
    label.classList.add('row-ix');
    label.innerText = index === 1 ? '' : `e_${index + 1}`;
    return label;
}
// Helper for numeric multi-inputs (e.g., int-mult, float-mult)
function addVariableNumericMultipleInputs(container, key, paramData, index=null) {
    const inputRow = document.createElement('div');
    inputRow.classList.add('input-row');
    inputRow.appendChild(createNumericInput(key, paramData, index, 'min'));
    inputRow.appendChild(createNumericInput(key, paramData, index, 'max'));
    container.appendChild(inputRow);
}
// Helper for single numeric inputs (e.g., int, float)
function addNumericInput(container, key, paramData, index=null) {
    const inputRow = document.createElement('div');
    inputRow.classList.add('input-row');
    inputRow.appendChild(createNumericInput(key, paramData, index, 'min'));
    inputRow.appendChild(createNumericInput(key, paramData, index, 'max'));
    container.appendChild(inputRow);
}
// Helper for categorical multi-inputs (e.g., cat-mult)
function addCategoricalMultiInput(container, key, paramData, numColumns) {
    for (let i = 0; i < numColumns; i++) {
        const inputRow = createInputRow(`e_${i + 1}`);
        inputRow.appendChild(createSelectInput(key, paramData.possible_values, i));
        container.appendChild(inputRow);
    }
}
// Helper for single categorical inputs (e.g., cat)
function addCategoricalInput(container, key, paramData, index=null) {
    const inputRow = createInputRow('Select categories');
    inputRow.appendChild(createSelectInput(key, paramData.possible_values, index));
    container.appendChild(inputRow);
}

// Helper for array inputs
function addArrayInput(container, key, paramData, numColumns) {
    const dimensions = paramData.dimensions;
    for (let i = 0; i < numColumns; i++) {
        const inputRow = createInputRow(`e_${i + 1}`);
        dimensions.forEach((dim, dimIndex) => {
            inputRow.appendChild(createDimensionLabel(dim));
            inputRow.appendChild(createNumericInput(key, paramData, i, 'min', dimIndex));
            inputRow.appendChild(createNumericInput(key, paramData, i, 'max', dimIndex));
        });
        container.appendChild(inputRow);
    }
}
// Helper for boolean inputs
function addBooleanInput(container, key, index) {
    const inputRow = createInputRow(key, true);
    //inputRow.appendChild(inputRow);
    container.appendChild(inputRow);
}
function addComplexInput(container, key, paramData, index){
    const inputRow = document.createElement('div');
    inputRow.classList.add('input-row')
    const input = document.createElement('input');
    input.type = 'text'
    input.name = 'param_${key}'
    if (paramData.value !== null) {input.placeholder = paramData.value} else {input.placeholder = '${key}'};

    inputRow.appendChild(input)
    container.appendChild(inputRow)
}

// Create numeric inputs with min/max handling
//key, paramData, i, 'min')
function createNumericInput(keyOrId, paramData = null, index = null, bound = null, dimIndex = null, simpleValue = null, simpleMin = null, simpleMax = null) {
    const input = document.createElement('input');
    input.type = 'number';

    if (paramData) {
        // Use complex structure with key, index, bound, and optional dimIndex
        const key = keyOrId;
        input.id = dimIndex !== null
            ? `${key}_${paramData.dimensions[dimIndex]}_${bound}_${index}`
            : `${key}_${bound}${index !== null ? `_${index}` : ''}`;

        const valueKey = bound === 'min' ? 'min_value' : 'max_value';
        const boundValue = Array.isArray(paramData[valueKey])
            ? paramData[valueKey][dimIndex]
            : paramData[valueKey];

        input.value = boundValue || 0;
        input.min = paramData.min_value;
        input.max = paramData.max_value;
    } else {
        // Use simpler structure with id, value, min, and max
        input.id = keyOrId;
        input.value = simpleValue;
        input.min = simpleMin;
        input.max = simpleMax;
    }

    return input;
}

// Create dimension labels for array inputs
function createDimensionLabel(dim) {
    const dimLabel = document.createElement('div');
    dimLabel.classList.add('dim-label');
    dimLabel.innerText = `${dim}:`;
    return dimLabel;
}

async function updateFixedParameters(keyList, selData, container_name, append=false, index=null){
    const container = document.getElementById(container_name);
    const numColumnsInput = document.getElementById('num_electrodes');
    const numColumnsValue = append === false
        ? (numColumnsInput ? numColumnsInput.value : 0)
        : null;

    if (!append) container.innerHTML = ''; // Clear existing parameters

    const inputActions = {
        'int-mult': addFixedNumericMultipleInputs,
        'float-mult': addFixedNumericMultipleInputs,
        'cat-mult': addCatMultInputs,
        'int': addSingleNumericInput,
        'float': addSingleNumericInput,
        'cat': addSingleCatInput,
        'array': addArrayInputs,
        'boolean': addBooleanInput,
        //'complex-mult' : addComplexMultipleInputs
        'complex': addComplexInput,

    };

    keyList.forEach(key => {
        console.log(key)
        console.log(selData[key])
        const type = selData[key]['type'];
        const parameterContainer = createParameterContainer();
        const label = createLabel(key, type, selData[key]);
        const inputsContainer = document.createElement('div');
        inputsContainer.classList.add('inputs-container');

        const action = inputActions[type];
        if (action) {
            action(inputsContainer, key, selData[key], numColumnsValue);
        } else {
            console.warn(`Unknown type: ${type}`);
        }

        parameterContainer.appendChild(label);
        parameterContainer.appendChild(inputsContainer);
        container.appendChild(parameterContainer);
    });
}

async function updateVariableParameters(keyList, selData, container_name, append=false){
    const container = document.getElementById(container_name);
    const numColumnsInput = document.getElementById('num_electrodes');

    const numColumnsValue = append === false
        ? (numColumnsInput ? numColumnsInput.value : 0)
        : null;

    if (!append) container.innerHTML = ''; // Clear existing parameters

    const inputActions = {
        'int-mult': addVariableNumericMultipleInputs,
        'float-mult': addVariableNumericMultipleInputs,
        'int': addNumericInput,
        'float': addNumericInput,
        'cat-mult': addCategoricalMultiInput,
        'cat': addCategoricalInput,
        'array': addArrayInput,
        'boolean': addBooleanInput
    };

    const fixedAction = {
        'int-mult': addFixedNumericMultipleInputs,
        'float-mult': addFixedNumericMultipleInputs,
        'int': addSingleNumericInput,
        'float': addSingleNumericInput,
        'cat-mult': addCatMultInputs,
        'cat': addSingleCatInput,
        'array': addArrayInputs,
        'boolean': addBooleanInput
    };

    keyList.forEach(key => {
        const type = selData[key]['type'];
        const paramData = selData[key];
        const parameterContainer = createParameterContainer(key);
        const label = createLabel(key, type, selData[key]);
        const inputsContainer = document.createElement('div');
        inputsContainer.classList.add('inputs-container');

        // Add checkbox for min-max toggle
        const toggleContainer = document.createElement('div');
        const toggleLabel = document.createElement('label');
        toggleLabel.textContent = "Fix value";
        toggleLabel.classList.add('white-text');

        const toggleCheckbox = document.createElement('input');
        toggleCheckbox.type = 'checkbox';

        toggleCheckbox.id = `toggle_${key}`;

        toggleContainer.appendChild(toggleLabel);
        toggleContainer.appendChild(toggleCheckbox);

        // Add event listener to handle toggle
        toggleCheckbox.addEventListener('change', (event) => {
            const isFixed = event.target.checked;
            inputsContainer.innerHTML = ''; // Clear existing inputs
            const action = isFixed ? fixedAction[type] : inputActions[type];

            if (action) {
                action(inputsContainer, key, paramData);
            } else {
                console.warn(`Unknown type: ${type}`);
            }
        });

        const initialAction = toggleCheckbox.checked ? fixedAction[type] : inputActions[type];

        if (initialAction) {
            initialAction(inputsContainer, key, paramData);
        } else {
            console.warn(`Unknown type: ${type}`);
        }

        parameterContainer.appendChild(label);
        parameterContainer.appendChild(toggleContainer); // Add the toggle checkbox
        parameterContainer.appendChild(inputsContainer);
        container.appendChild(parameterContainer);
    });
}

// Function to draw the plot
function drawPlot() {
    const num_el_container = document.getElementById('num_electrodes')
    if (!num_el_container){ //There are no electrodes in the problem, this is not needed.
        return
    }
    const num_el = parseInt(num_el_container.value, 10)
    const centerX = 110
    const canvas = document.getElementById('plotCanvas')

    let height
    if (globalData['experiment']['name'].startsWith("axonsim")){
        height = parseFloat(document.getElementById('dia').value)
    }
    else if (globalData['experiment']['name'].startsWith("cajal")){
        height = parseFloat(document.getElementById('axon_diameter').value)
    }
    else{
        console.error("The input is not valid")
    }

    const dims = globalData['experiment']['e_pos']['dimensions']
    const ctx = canvas.getContext('2d');
    // Clear the canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw the x and y axes
    ctx.beginPath();
    ctx.moveTo(0, canvas.height / 2); // X axis
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.moveTo(canvas.width / 2, 0); // Y axis
    ctx.lineTo(canvas.width / 2, canvas.height);
    ctx.stroke();

    // Draw the rectangle
    const rectWidth = 250; // Adjust rectangle width
    const rectX = centerX - rectWidth / 2;
    const rectY = (canvas.height / 2) - (height / 2); // Centered vertically

    ctx.fillStyle = 'rgba(100, 150, 200, 0.5)'; // Rectangle color
    ctx.fillRect(rectX, rectY, rectWidth, height);
    ctx.fillStyle = 'red'; // Electrode marker color

    // Add electrode positions
    for (let n=0; n<=num_el-1; n++){
        let pos = []
        for (let dim in dims) {
            const el = document.getElementById(`e_pos_${dims[dim]}_${n}`);
            pos.push(parseFloat(el.value))
        }
        const markerX = 10*pos[0] + (canvas.width / 2); // Center the electrodes
        const markerY = -10*pos[1] + (canvas.height/2); // Visually speaking, Positive values are above from the origin
        ctx.fillText('x', markerX, markerY); // Place above rectangle
    }
}

function draw3DPlot() {
    const num_el_container = document.getElementById('num_electrodes')
    if (!num_el_container) { // There are no electrodes in the problem, this is not needed.
        return
    }
    const num_el = parseInt(num_el_container.value, 10)
    const centerX = 110
    const canvas = document.getElementById('plotCanvas')
    const height = parseFloat(document.getElementById('axon_diameter').value)
    const dims = globalData['experiment']['e_pos']['dimensions']

    // Create the scene and camera
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(75, canvas.width / canvas.height, 0.1, 1000)
    const renderer = new THREE.WebGLRenderer({ canvas: canvas })
    renderer.setSize(canvas.width, canvas.height)

    // Set camera position
    camera.position.z = 500

    // Add axes for reference
    const axesHelper = new THREE.AxesHelper(100)
    scene.add(axesHelper)

    // Draw the rectangle as a 3D box
    const geometry = new THREE.BoxGeometry(250, height, 10) // Width, height, depth
    const material = new THREE.MeshBasicMaterial({ color: 0x6496c8, transparent: true, opacity: 0.5 })
    const rect = new THREE.Mesh(geometry, material)
    scene.add(rect)

    // Center the rectangle
    rect.position.set(centerX, 0, 0) // Adjust rectangle position

    // Create a material for the electrode markers
    const electrodeMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000 })

    // Electrode positions
    for (let n = 0; n <= num_el - 1; n++) {
        let pos = []
        for (let dim in dims) {
            const el = document.getElementById(`e_pos_${dims[dim]}_${n}`);
            pos.push(parseFloat(el.value))
        }
        const electrodeGeometry = new THREE.SphereGeometry(5, 32, 32) // Small sphere for electrodes
        const electrode = new THREE.Mesh(electrodeGeometry, electrodeMaterial)

        // Set electrode position (scaled to fit canvas)
        electrode.position.set(10 * pos[0] + centerX, -10 * pos[1], 10 * pos[2])

        scene.add(electrode)
    }

    // Rendering loop
    function animate() {
        requestAnimationFrame(animate)
        renderer.render(scene, camera)
    }
    animate() // Start animation loop
}

function renderJSON(container, data) {
    for (const [key, value] of Object.entries(data)) {
        const keyElement = document.createElement('div');
        keyElement.className = 'json-key';
        keyElement.textContent = `${key}: `;

        const valueElement = document.createElement('span');
        valueElement.className = 'json-value';
        renderJSON(valueElement, value); // Recursive call
        keyElement.appendChild(valueElement);
        div.appendChild(keyElement);

    }
}

function updateJSONDisplay() {
    const parameters = collectInputs('sidebar-left', 'pulse_');
    const structuredData = restructureParameters(parameters);
    const jsonDisplay = document.getElementById('jsonDisplay');
    jsonDisplay.innerHTML = ''; // Clear previous content
    renderJSON(jsonDisplay, structuredData);
}
async function fetchExperimentsAndModels() {
    await fetchData('experiments', 'experiment', 'Select Experiment');
    await fetchData('models', 'model', 'Select Model');
    // await fetchData('optimization', 'optimization', 'Select Model');
}

async function initializeOptimizerParameters() {
    console.log("Initializing static parameters...");
    // Load optimizer parameters once
    await loadParameters("optimizer", { selectable: false });
    console.log("Initialization complete.");
}

async function createIframes() {
    await removeIframes();
    const iframeContainer = document.getElementById('iframeContainer');

    // iFrame configurations for differnet plot types
    const iframeConfigs = [
        { id: 'mainPlot', src: `http://localhost:${APP_CONFIG.port}/plot?type=plotly` },
        { id: 'scatterPlot', src: `http://localhost:${APP_CONFIG.port}/plot?type=contour`},
    //    { id: 'pulsesPlot', src: `http://localhost:${APP_CONFIG.port}/plot?type=pulses`},
    ];

    iframeConfigs.forEach(({ id, src }) => {
        if (!iframeContainer.querySelector(`#${id}`)) { // Scope lookup to iframeContainer
            const iframe = document.createElement('iframe');
            iframe.id = id;
            iframe.src = src;
            iframeContainer.appendChild(iframe);
        }
    });
    console.log("iFrames Created succesfully")
}

// Function to remove (or hide) iFrames
function removeIframes() {
    const iframeContainer = document.getElementById('iframeContainer');
    if (!iframeContainer) return; // Ensure iframeContainer exists before proceeding

    let existingIframes = iframeContainer.querySelectorAll('iframe');

    while (existingIframes.length > 0) {
        existingIframes[0].remove();
    }
}


document.addEventListener("DOMContentLoaded", () => {
const selectFolderButton = document.getElementById("selectFolder");
const folderPathDisplay = document.getElementById("folderPath");

if (selectFolderButton) {
    selectFolderButton.addEventListener("click", async () => {
        try {
            const handle = await window.showDirectoryPicker();
            folderPathDisplay.textContent = handle.name; // Show selected folder name
            window.selectedFolderHandle = handle; // Store handle for later use
        } catch (err) {
            console.error("Folder selection cancelled or failed", err);
        }
    });
}
document.querySelector('.right-toggle').addEventListener('click', () => {
    toggleSidebar('.sidebar-right');
});

document.querySelector('.left-toggle').addEventListener('click', () => {
    toggleSidebar('.sidebar');
});
});

document.addEventListener('DOMContentLoaded', fetchExperimentsAndModels)
document.addEventListener("DOMContentLoaded", () => {initializeOptimizerParameters();})

// document.getElementById('sidebar-left').addEventListener('input', event => {
//     if (event.target.tagName === 'INPUT') {
//         updateJSONDisplay();
//     }
// });

// document.getElementById('sidebar-right').addEventListener('input', event => {
//     if (event.target.tagName === 'INPUT') {
//         updateJSONDisplay();
//     }
// });

// Save folder selection
const folderInput = document.getElementById('selectFolder');
const folderPath = document.getElementById('folderPath');

folderInput.addEventListener('change', () => {
    if (folderInput.files.length > 0) {
        // Get only the folder name, ignore files
        const folderName = folderInput.files[0].webkitRelativePath.split("/")[0];
        folderPath.textContent = folderName;
    } else {
        folderPath.textContent = "No folder selected";
    }
});
// Load file selection
const fileInput = document.getElementById('inputFile');
const filePath = document.getElementById('filePath');

fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
        // Show selected file name
        filePath.textContent = fileInput.files[0].name;
    } else {
        filePath.textContent = "No file selected";
    }
});
