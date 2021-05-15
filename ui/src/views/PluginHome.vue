<template>
<div>

    <b-modal id="modal-logviewer" title="Plugin Manager Logs" size="lg">
        <LogViewer :nocard="true" :logStreamUrl="`${appConfig.API_URL}/api/stream/scoped/plugin_manager`"> </LogViewer>
    </b-modal>

    <b-navbar toggleable="lg" type="light" variant="alert">
        Status:
        <b-button-group>
            <b-button :disabled="selectedStatus==''" size="sm" variant="primary" @click="updateStatusFilter('')">Any</b-button>
            <b-button :disabled="selectedStatus=='installed'" size="sm" variant="primary" @click="updateStatusFilter('installed')">Installed</b-button>
            <b-button :disabled="selectedStatus=='avail'" size="sm" variant="primary" @click="updateStatusFilter('avail')">Not Installed</b-button>

        </b-button-group>
        <b-form-input v-model="searchtext" placeholder="Search Plugins" style="width:250px;" class='ml-2'></b-form-input>
        <!-- <span class="mr-2 ml-5">Category Filter:</span> -->
        <!-- <b-button-group size="sm">
            <b-button class="mr-2" v-for="(btn, idx) in filterCategories" :key="idx" :pressed.sync="btn.state"  @click="updateList()" variant="info" pill>{{ btn.caption }} </b-button>
   
        </b-button-group> -->
        <b-button class="ml-auto mr-2" v-b-modal:modal-logviewer size="sm" variant="info"><i class="fa fa-bug"></i> View Logs</b-button>

    </b-navbar>
    <b-navbar toggleable="lg" type="light" variant="alert">

        <b-form-checkbox class="ml-2" switch v-model="onlyWorkspacePlugins">Show only Workspace Plugins</b-form-checkbox>

    </b-navbar>

    <div class="mt-4"></div>

    <div v-if="Object.keys(filteredPlugins).length >0">
        <b-container fluid>
            <div v-for="(item, idx) in filteredPlugins" :key="idx">
                <b-row>
                    <b-col>
                        <div>
                            <h4>
                                {{item.name}}
                            </h4>
                        </div>
                    </b-col>
                    <b-col>
                        <span v-if="item.source.name == 'Workspace'">
                            <b-badge pill variant="warning"><i class="fa fa-code"></i> Workspace Plugin</b-badge>
                        </span>

                    </b-col>
                    <b-col>
                        <div>
                            <b-button v-if="item.status == 'AVAILABLE'" size="sm" variant="info" @click="installPlugin(item.id,item.version);">Install</b-button>
                            <b-button v-else-if="item.status == 'INSTALLING'" size="sm" variant="primary" disabled>
                                <b-spinner small type="grow"></b-spinner>Installing...
                            </b-button>
                            <div v-else>
                                <b-button v-if="item.status == 'INSTALL_FAILED'" size="sm" variant="secondary" class="mr-2" @click="installPlugin(item.id,item.version);">Retry Install</b-button>
                                <b-button v-if="item.status == 'INSTALLED'" size="sm" variant="secondary" class="mr-2" @click="installPlugin(item.id,item.version);">Reinstall</b-button>

                                <b-button class="mr-2" size="sm" variant="danger" @click="removePlugin(item.id,item.version)">Uninstall</b-button>
                                <!-- <b-button v-if="['INSTALLED','PREPPED_RELOAD'].includes(item.status)" class="mr-2" size="sm" variant="secondary" @click="reloadPlugin(item.id,item.version)">Reload</b-button> -->
                            </div>
                        </div>
                    </b-col>
                </b-row>
                                <b-row>
                    <b-col>

                        <div class="ml-2">
                            <p>{{item.description}}</p>
                        </div>
                    </b-col>
                </b-row>
                <div class="mt-1"></div>
                <b-row>
                    <b-col class="">

                        <div>
                            <span class="data_label mt-1">ID: </span><span>{{item.id}}</span>
                        </div>
                        <div>
                            <span class="data_label mt-1">Version: </span><span>{{item.version}}</span>
                        </div>
                        <div>
                            <span class="data_label mt-1">Categories: </span>
                            <span>{{item.categories.join(",")}}</span>
                        </div>
                    </b-col>
                    <b-col class="">
                        <div>
                            <span class="data_label mt-1">Source: </span>
                            <span>{{item.source.name}}</span>
                        </div>
                        <div>
                            <span class="data_label mt-1">Original Author: </span>
                            <span>{{item.original_author}}</span>
                        </div>
                        <div>
                            <span class="data_label mt-1">Plugin Author: </span>
                            <span>{{item.plugin_author}}</span>
                        </div>
                    </b-col>
                    <b-col class="">
                        <span class="data_label mt-1">State: </span>
                        <span v-if="item.status == 'PREPPED_RELOAD'" style="color:yellow">**Restart piSTAR Lab to complete installation*</span>
                        <span v-else>{{item.status}} </span>
                        <span v-if="(item.status == 'INSTALL_FAILED' || item.status == 'UNINSTALL_FAILED') && item.status_msg">
                            <pre>{{item.status_msg}}</pre>
                        </span>
                    </b-col>
                </b-row>

                <span v-if="idx != Object.keys(filteredPlugins).length - 1">
                    <hr /> </span>

            </div>
        </b-container>
    </div>

    <div v-else>

        No Plugins Found

    </div>

</div>
</template>

<script>
// @ is an alias to /src
import axios from "axios";
import {
    appConfig
} from "../app.config";
import gql from 'graphql-tag'
import {
    timedelta,
    timepretty
} from "../funcs";

const fields = [{
        key: "link",
        label: "Plugin UID",
        sortable: true,
    },

    {
        key: "desc.tags",
        label: "Tags",
        sortable: true,
        formatter: (v) => {
            if (!v) return "";
            return v.join(", ");
        },
    },

    {
        key: "info.creation_time",
        label: "Creation Time",
        sortable: true,
        formatter: timedelta,
    },
    {
        key: "job_data.state",
        label: "Job State",
    },
];
import LogViewer from "../components/LogViewer.vue";

export default {
    name: "PluginHome",
    components: {
        LogViewer
    },
    props :{
        showWorkspacePlugins:Boolean
    },
    data() {
        return {
            appConfig,
            searchtext: "",
            fields: fields,
            allPlugins: {},
            error: "",
            selected: [],

            selectedStatus: "",
            onlyWorkspacePlugins: false,
            filterCategories: {
                agents: {
                    caption: "Agents",
                    state: false
                },
                envs: {
                    caption: "Enivonrments",
                    state: false
                },
                tasks: {
                    caption: "Tasks",
                    state: false
                },
            },
        };
    },

    computed: {
        filteredPlugins() {

            if (this.selectedStatus == 'installed') {
                return this.filtered.filter(plugin => {
                    return plugin.status != "AVAILABLE";
                })

            } else if (this.selectedStatus == 'avail') {
                return this.filtered.filter(plugin => {
                    return plugin.status == "AVAILABLE" || plugin.status == "INSTALLING" || plugin.status == "INSTALL_FAILED" || plugin.status == "UNINSTALL_FAILED";
                })
            } else {
                return this.filtered
            }
        },

        filtered() {
            let categories = new Set();

            Object.keys(this.filterCategories).forEach(cat => {
                if (this.filterCategories[cat]["state"]) {
                    categories.add(cat);
                }
            });

            const filteredList = []
            Object.values(this.allPlugins).forEach((plugin) => {
                let include = false;

                if (this.onlyWorkspacePlugins && plugin['source']['name'] != "Workspace") {
                    return
                }
                for (const cat of plugin.categories) {
                    if (categories.has(cat)) {
                        include = true
                        break;
                    }
                }
                if (plugin.categories.length == 0 || (plugin.categories.length == this.filterCategories.length) || include) {

                    filteredList.push(plugin)
                }

            })
            if (this.searchtext != "") {
                return filteredList.filter((v) => {
                    return v.name.toLowerCase().includes(this.searchtext.toLowerCase())
                })
            } else {
                return filteredList
            }
        },

    },
    methods: {
        getPluginKey(pluginId, pluginVersion) {
            return `${pluginId}__v${pluginVersion}`
        },


        updateList(btn) {
            this.loadData();

        },
        installPlugin(pluginId, pluginVersion) {
            console.log("Installing " + this.getPluginKey(pluginId, pluginVersion));
            const plugin = this.allPlugins[this.getPluginKey(pluginId, pluginVersion)]
            plugin.status = "INSTALLING"
            this.$set(this.allPlugins, this.getPluginKey(pluginId, pluginVersion), plugin)

            axios
                .get(`${appConfig.API_URL}/api/plugins/action/install/${pluginId}/${pluginVersion}`)
                .then((response) => {
                    this.loadData();
                    this.updateStatusFilter('installed')
                })
                .catch(function (error) {
                    this.errorMessage = error;
                    this.loadData();
                });

        },
        removePlugin(pluginId, pluginVersion) {
            console.log("Uninstall " + this.getPluginKey(pluginId, pluginVersion));
            this.allPlugins[this.getPluginKey(pluginId, pluginVersion)].status = "REMOVING"

            axios
                .get(`${appConfig.API_URL}/api/plugins/action/uninstall/${pluginId}/${pluginVersion}`)
                .then((response) => {
                    this.errorMessage = JSON.stringify(response.data);
                    this.loadData();
                })
                .catch(function (error) {
                    this.errorMessage = error;
                });

            this.loadData();
        },
        reloadPlugin(pluginId, pluginVersion) {
            console.log("Reloading " + this.getPluginKey(pluginId, pluginVersion));

            axios
                .get(`${appConfig.API_URL}/api/plugins/action/reload/${pluginId}/${pluginVersion}`)
                .then((response) => {
                    this.errorMessage = JSON.stringify(response.data);
                    this.loadData();
                })
                .catch(function (error) {
                    this.errorMessage = error;
                });
            this.loadData();
        },
        loadData() {

            axios
                .get(`${appConfig.API_URL}/api/plugins/list`)
                .then((response) => {
                    this.allPlugins = response.data["items"]

                })
                .catch((e) => {
                    this.error = e;
                });
        },
        updateStatusFilter(status) {
            console.log(status)
            this.selectedStatus = status
        }
    },
    // Fetches posts when the component is created.
    created() {
        console.log("HI")
        if (this.showWorkspacePlugins){
            this.onlyWorkspacePlugins=true
        }
        // if (this.category) {
        //     console.log(this.category);
        //     this.filterCategories[this.category]["state"] = true;
        // } else {
        Object.keys(this.filterCategories).forEach((cat) => {
            this.filterCategories[cat]["state"] = true;
        });
        // }
        this.loadData();
    },
};
</script>

<style>
/* a.page-link{
   color: black;
 } */
</style>
