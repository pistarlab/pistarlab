<template lang="html">
<div>

    <div v-if="!editorMode && $apollo.queries.componentSpecs&& !$apollo.queries.componentSpecs.loading">
        <b-button-toolbar>
            <b-form-checkbox switch v-model="advancedMode"> Show All (Advanced)</b-form-checkbox>
            <b-button pill size="sm" class="ml-auto" variant="outline-secondary" @click="enableEditor()">View in Editor</b-button>
        </b-button-toolbar>

        <div class="mt-4"></div>

        <div>
            <div v-for="(data,i) in paramList" :key="i">
                <div v-if="skipEntry(data)">
                    <div class="d-flex flex-row">
                        <div v-for="n in data.depth" :key="n" class="ml-2 " style=" width:50px">
                        </div>
                        <div class="flex-column">
                            <div>
                                <label v-bind:for="`item_`+data.full_path">
                                    <span v-if="data.data_type=='component'">
                                        <i class="fas fa-caret-square-right"></i>

                                        {{data.displayed_name}}:
                                    </span>
                                    <span v-else>
                                        {{data.displayed_name}}:
                                    </span>
                                </label>
                            </div>
                            <div class="ml-3 row">
                                <div v-if="data.data_type=='component'">
                                    <select v-model="paramValues[data.full_path]" @change="updateParamList()" v-bind:id="`item_`+data.full_path">
                                        <option value="">-select-</option>
                                        <option v-for="(item,componentKey) in data.options" :key="componentKey" :value="componentKey">
                                            {{componentKey}}
                                        </option>
                                    </select>
                                </div>
                                <div v-else-if="data.data_type=='list'">
                                    {{displayList(paramValues[data.full_path])}}
                                    <b-modal v-bind:id="`modal_`+data.full_path">

                                        <b-textarea v-model="paramValues[data.full_path]">
                                        </b-textarea>
                                    </b-modal>
                                    <b-button variant="white" size="sm" v-b-modal="'modal_'+data.full_path"><i class="fa fa-edit"></i></b-button>
                                </div>
                                <div v-else-if="data.data_type=='dict'">
                                    {{paramValues[data.full_path]}}
                                    <b-modal v-bind:id="`modal_`+data.full_path">
                                        <b-textarea v-model="paramValues[data.full_path]">
                                        </b-textarea>
                                    </b-modal>
                                    <b-button variant="white" size="sm" v-b-modal="'modal_'+data.full_path"><i class="fa fa-edit"></i></b-button>
                                </div>

                                <div v-else-if="data.data_type=='category'">
                                    <span v-for="(optionval,oi) in data.type_info.options" :key="oi" class="mr-2">
                                        <input type="radio" v-bind:value="optionval" v-model="paramValues[data.full_path]" />
                                        <label class="ml-1"> {{optionval}}</label>
                                    </span>
                                </div>
                                <div v-else-if="data.data_type=='bool'">
                                    <span class="mr-2">
                                        <b-form-checkbox switch v-model="paramValues[data.full_path]"> {{paramValues[data.full_path]}}</b-form-checkbox>
                                    </span>
                                </div>

                                <div v-else-if="data.data_type=='float'" class="col-xs-4">
                                    <b-input v-model.number="paramValues[data.full_path]"></b-input>
                                    <span v-if="data.type_info.use_range">
                                        <input type="range" class="form-control-range mt-2" v-bind:step="(data.type_info.max-data.type_info.min)/10000" v-bind:min="data.type_info.min" v-bind:max="data.type_info.max" v-model.number="paramValues[data.full_path]">
                                    </span>
                                </div>
                                <div v-else-if="data.data_type=='int'" class="col-xs-4">
                                    <b-input v-model.number="paramValues[data.full_path]" type="number" v-bind:value="parseInt(paramValues[data.full_path])"></b-input>
                                    <span v-if="data.type_info.use_range">
                                        <input type="range" class="form-control-range mt-2" v-bind:step="Math.round((data.type_info.max-data.type_info.min)/1000)" v-bind:min="data.type_info.min" v-bind:max="data.type_info.max" v-model.number="paramValues[data.full_path]">
                                    </span>
                                </div>
                                <div v-else>
                                    <b-input v-model="paramValues[data.full_path]"></b-input>

                                </div>
                            </div>
                            <div v-if="data.description" class="ml-3 form-text text-muted small">
                                {{data.description}}
                            </div>
                        </div>

                    </div>

                    <hr />
                </div>
            </div>

        </div>

    </div>
    <div v-else>
        <b-button-toolbar>
            <b-button class="ml-auto" pill @click="enableFormMode()" size="sm" variant="outline-secondary">View in Form</b-button>
        </b-button-toolbar>

        <div class="mt-4"></div>

        <b-textarea v-model="paramValuesEditor" cols=200 rows=40></b-textarea>
        <!-- <editor v-model="paramValuesEditor" @init="editorInit" lang="json" width="100%" height="600"></editor> -->

    </div>
</div>
</template>

<script>
//USING https://github.com/chairuosen/vue2-ace-editor

import gql from "graphql-tag";

var flattenObject = function (ob, validKeys, parentPath = null) {
    var toReturn = {};

    for (var i in ob) {
        if (!Object.prototype.hasOwnProperty.call(ob, i)) continue;

        var fullPath = i
        if (parentPath) {
            fullPath = parentPath + "." + i
        }

        if ((typeof ob[i]) == 'object') {
            if (validKeys.has(fullPath)) {
                var flatObject = flattenObject(ob[i], validKeys, fullPath);
                for (var x in flatObject) {
                    if (!Object.prototype.hasOwnProperty.call(flatObject, x)) continue;

                    toReturn[i + '.' + x] = flatObject[x];
                }
            } else {
                toReturn[i] = JSON.stringify(ob[i], null, 2)
            }
        } else {
            toReturn[i] = ob[i];
        }
    }
    return toReturn;
};

function assign(obj, keyPath, value) {
    var lastKeyIndex = keyPath.length - 1;
    for (var i = 0; i < lastKeyIndex; ++i) {
        var key = keyPath[i];
        if (!(key in obj)) {
            obj[key] = {}
        }
        obj = obj[key];
    }
    obj[keyPath[lastKeyIndex]] = value;
}

const GET_ALL_COMPONENTS = gql `query
{
  componentSpecs: allComponentSpecs {
    edges {
      node {
        ident
        created
        category
        parentClassEntryPoint
        extensionId
        config
        params
      }
    }
  }
}
`;

export default {
    components: {
        // editor: require('vue2-ace-editor'),
    },
    apollo: {
        // Simple query that will update the 'hello' vue property
        componentSpecs: {
            query: GET_ALL_COMPONENTS,
            // Optional result hook
            result({
                data,
                loading,
                networkStatus
            }) {
                console.log(this.componentGroups)
                this.updateParamList(true)
            },
        },
    },
    data() {
        return {
            editorMode: false,
            componentSpecs: null,
            advancedMode: false,
            paramValues: {},
            paramValuesEditor: "{}",
            paramList: [],
        };
    },
    watch: {
        paramValues: {
            handler: function (val,oldVal) {
                this.paramValues = val
                this.save()
            },
            deep: true
        },
        paramValuesEditor: function (val) {
                if (this.editorMode) {
                    this.paramValuesEditor = val
                    this.save()
                }
            }
        
    },

    props: {
        params: Object,
        values: Object,
        buttonText: String,
        interfaceFilter: String
    },
    computed: {

        componentGroups() {
            console.log("--ERE")

            if (this.componentSpecs == null) return {}

            let componentGroups = {};
            //this.allAgents.edges.map((edge)=>edge.node)
            this.componentSpecs.edges.forEach(edge => {
                let spec = edge.node
                if (!(spec.parentClassEntryPoint in componentGroups)) {
                    componentGroups[spec.parentClassEntryPoint] = {}
                }
                var config = {}
                try {
                    config = JSON.parse(spec.config)
                } catch (err) {
                    //
                    console.log(`error parsing ${spec.specId}`)
                }
                var params = JSON.parse(spec.params)
                console.log(JSON.stringify(params, null, 2))

                componentGroups[spec.parentClassEntryPoint][spec.ident] = {
                    'params': params
                }

            })
            return componentGroups

        }
    },
    methods: {
        skipEntry(data) {
            if (this.advancedMode) {
                return true
            }
            else if (data.mode == "default") {
                if (!this.interfaceFilter || this.interfaceFilter == "")
                    return true
                else if (data.interfaces.includes(this.interfaceFilter))
                    return true
            }
            return false

        },
        enableEditor() {

            this.updateParamList()
            let paramsNested = this.getTypedParamValues()
            this.paramValuesEditor = JSON.stringify(paramsNested, null, 2)
            this.editorMode = true
        },
        enableFormMode() {
            this.loadParams(JSON.parse(this.paramValuesEditor))
            this.editorMode = false

        },
        displayList(s) {
            try {
                return JSON.parse(s).join(", ")
            } catch (error) {
                console.log(`error parsing ${s}, error =  ${error}`)
                return ""
            }

        },
        loadParams(paramsToLoad) {
            let validKeys = new Set()
            this.paramList.forEach((param) => {
                let key = param.full_path
                let keyParts = key.split('.')

                for (var i = 0; i < keyParts.length; i++) {
                    let subKey = keyParts.slice(0, i).join(".")
                    validKeys.add(subKey)

                }

            })

            this.paramValues = {}
            var flattenedObject = flattenObject(paramsToLoad, validKeys, null)
            this.paramValues = flattenedObject

        },
        getParamListForComponent(inParams, parentPath = null, depth = 0) {
            console.log("Step:" + parentPath)
            let params = JSON.parse(JSON.stringify(inParams))
            let allParams = []

            Object.keys(params).forEach(key => {
                let param = params[key]
                if (parentPath != null) {
                    param.full_path = parentPath + "." + param.path
                } else {
                    param.full_path = param.path
                }
                param.depth = depth
                if (param.data_type == "component" && param.component_type in this.componentGroups && param.full_path.endsWith("__component_id")) {
                    param.options = JSON.parse(JSON.stringify(this.componentGroups[param.component_type]))
                    allParams.push(param)
                    if (param.full_path in this.paramValues) {
                        let selectedComponentValue = this.paramValues[param.full_path]
                        if (selectedComponentValue in this.componentGroups[param.component_type]) {
                            let componentParams = JSON.parse(JSON.stringify(this.componentGroups[param.component_type][selectedComponentValue].params));
                            let pathparts = param.full_path.split(".")
                            let componentPath = pathparts.slice(0, pathparts.length - 1).join(".")
                            let expandedParams = this.getParamListForComponent(componentParams, componentPath, depth + 1)
                            expandedParams.forEach((v) => {
                                allParams.push(v)
                            })
                        }
                    }
                } else {
                    allParams.push(param)
                }
            });
            return allParams
        },
        updateParamList(firstCall = false) {
            console.log("Updating")
            this.paramList = []

            let allParams = []
            let done = false
            let allPaths = new Set()
            while (!done) {
                done = true
                allParams = this.getParamListForComponent(this.params)
                allPaths = new Set(allParams.map((v) => {
                    if (!(v.full_path in this.paramValues)) {
                        this.$set(this.paramValues, v.full_path, v.default)
                        if (v.data_type == "component" && v.default) {
                            done = false

                        }
                    }
                    return v.full_path
                }))
            }
            this.paramList = allParams;

            Object.keys(this.paramValues).forEach((v) => {
                if (!allPaths.has(v)) {
                    delete this.paramValues[v]

                }

            })
            if (firstCall && Object.keys(this.values).length > 0) {

                this.loadParams(this.values)
            }

        },
        getTypedParamValues() {

            let resultParams = {}

            this.paramList.forEach((p) => {

                let pathParts = p.full_path.split(".")

                let paramValue = this.paramValues[p.full_path];
                let preppedVal = null;
                if (p.data_type == "list" || p.data_type == "dict") {
                    try {
                        preppedVal = JSON.parse(paramValue);
                    } catch (error) {
                        preppedVal = null
                    }

                } else if (p.data_type == 'int') {
                    preppedVal = parseInt(paramValue)
                } else if (p.data_type == 'float') {
                    preppedVal = parseFloat(paramValue)
                } else {
                    preppedVal = paramValue
                }

                assign(resultParams, pathParts, preppedVal)

            })
            return resultParams

        },

        save() {
            if (this.editorMode) {
                this.updateParamList()
                let paramsNested = JSON.parse(this.paramValuesEditor)
                this.paramValuesEditor = "{}"
                this.$emit('update', paramsNested)
            } else {
                this.updateParamList()
                let paramsNested = this.getTypedParamValues()
                this.$emit('update', paramsNested)
            }
        },

        // editorInit: function () {
        //     require('brace/ext/language_tools') //language extension prerequsite...
        //     require('brace/mode/html')
        //     require('brace/mode/json') //language
        //     require('brace/mode/less')
        //     require('brace/theme/twilight')
        //     require('brace/theme/chrome')

        //     require('brace/snippets/javascript') //snippet
        // },
        cancel() {
            //          
        },
        onError() {
            //
        },
    },

    created() {
        //

    },
};
</script>

<style scoped>
.ace_editor {
    font-size: 16px;
}
</style>
