<template>
<div>

    <b-modal id="modal-logviewer" title="Plugin Manager Logs" size="lg">
        <LogViewer :nocard="true" :logStreamUrl="`${appConfig.API_URL}/api/stream/scoped/plugin_manager`"> </LogViewer>
    </b-modal>
    <b-modal id="modal-create-plugin" title="Create New Plugin" size="lg" @ok="createNewPlugin()">
        <p>
            Create a new plugin in your workspace.
        </p>
        <div class="mt-2"></div>
        <label for="newPluginId">Plugin Id:</label>
        <b-form-input id="newPluginId" v-model="enteredPluginId" trim></b-form-input> {{newPluginId}}
        <div class="mt-1"></div>
        <label for="newPluginName">Plugin Name:</label>
        <b-form-input id="newPluginName" v-model="newPluginName" trim></b-form-input>
        <label for="newPluginDescription">Description:</label>
        <b-form-input id="newPluginDescription" v-model="newPluginDescription" trim></b-form-input>
    </b-modal>
    <b-navbar toggleable="lg" type="light" variant="alert">
        <b-button-toolbar>
            <b-button :disabled="selectedStatus=='installed'" size="sm" variant="primary" @click="updateStatusFilter('installed')">Current Plugins</b-button>
            <b-button class="ml-2" :disabled="selectedStatus=='avail'" size="sm" variant="primary" @click="updateStatusFilter('avail')">Install Plugins</b-button>

        </b-button-toolbar>
        <!-- <span class="mr-2 ml-5">Category Filter:</span> -->
        <!-- <b-button-group size="sm">
            <b-button class="mr-2" v-for="(btn, idx) in filterCategories" :key="idx" :pressed.sync="btn.state"  @click="updateList()" variant="info" pill>{{ btn.caption }} </b-button>
   
        </b-button-group> -->
        <b-button class="ml-auto mr-2" v-b-modal:modal-logviewer size="sm" variant="info"><i class="fa fa-bug"></i> View Logs</b-button>

    </b-navbar>
    <b-navbar toggleable="lg" type="light" variant="alert">

        <b-form-checkbox class="ml-2" switch v-model="onlyWorkspacePlugins">Show only plugins in Workspace</b-form-checkbox>

        <b-button class="ml-auto" v-b-modal:modal-create-plugin size="sm" variant="success"><i class="fa fa-plus"></i> Create New Plugin</b-button>
    </b-navbar>

    <div class="mt-4"></div>
    <div v-if="Object.keys(filteredPlugins).length >0">
        <b-container fluid>
            <div v-for="(item, idx) in filteredPlugins" :key="idx">
                <b-row>
                    <b-col>
                        <div>

                            {{item.name}}
                        </div>
                    </b-col>
                    <b-col>
                        <span v-if="item.source.name == 'Workspace'">
                            <b-badge pill variant="warning"><i class="fa fa-code"></i> This plugin is in your workspace </b-badge>
                        </span>

                    </b-col>
                    <b-col>
                        <div>
                            <b-button v-if="item.status == 'AVAILABLE'" size="sm" variant="outline-primary" @click="installPlugin(item.id,item.version);">Install</b-button>
                            <b-button v-else-if="item.status == 'INSTALLING'" size="sm" variant="outline-primary" disabled>
                                <b-spinner small type="grow"></b-spinner>Installing...
                            </b-button>
                            <div v-else>
                                <b-button v-if="item.status == 'INSTALL_FAILED'" size="sm" variant="outline-primary" class="mr-2" @click="installPlugin(item.id,item.version);">Retry Install</b-button>
                                <b-button v-if="item.status == 'INSTALLED'" size="sm" variant="outline-primary" class="mr-2" @click="installPlugin(item.id,item.version);">Reinstall</b-button>

                                <b-button class="mr-2" size="sm" variant="outline-danger" @click="removePlugin(item.id,item.version)">Uninstall</b-button>
                                <b-button v-if="['INSTALLED','PREPPED_RELOAD'].includes(item.status)" class="mr-2" size="sm" variant="outline-secondary" @click="reloadPlugin(item.id,item.version)">Reload</b-button>
                            </div>
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
                            <span class="data_label mt-1">Source Name: </span>
                            <span>{{item.source.name}}</span>
                        </div>
                        <div>
                            <span class="data_label mt-1">Author: </span>
                            <span>{{item.author}}</span>
                        </div>
                    </b-col>
                    <b-col class="">
                        <span class="data_label mt-1">State: </span>
                        <span v-if="item.status == 'PREPPED_RELOAD'">**Restart Required**</span>
                        <span v-else>{{item.status}} </span>
                        <span v-if="item.status_msg">
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
    data() {
        return {
            appConfig,
            searchQuery: "",
            fields: fields,
            allPlugins: {},
            error: "",
            selected: [],
            enteredPluginId: "",
            newPluginName: "",
            newPluginDescription: "",
            selectedStatus: "installed",
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

            } else {
                return this.filtered.filter(plugin => {
                    return plugin.status == "AVAILABLE" || plugin.status == "INSTALLING" || plugin.status == "INSTALL_FAILED" || plugin.status == "UNINSTALL_FAILED";
                })
            }
        },
        newPluginId() {
            return "pistarlab-" + this.enteredPluginId
        },

        filtered() {
            let categories = new Set();

            Object.keys(this.filterCategories).forEach(cat => {
                if (this.filterCategories[cat]["state"]) {
                    categories.add(cat);
                }
            });

            const filtered = []
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

                    filtered.push(plugin)
                }

            })
            return filtered
        },

    },
    methods: {
        getPluginKey(pluginId, pluginVersion) {
            return `${pluginId}__v${pluginVersion}`
        },
        createNewPlugin() {

            let outgoingData = {
                'plugin_id': this.newPluginId,
                'plugin_name': this.newPluginName,
                'description': this.newPluginDescription

            }
            axios
                .post(`${appConfig.API_URL}/api/plugin/create`, outgoingData)
                .then((response) => {
                    console.log(response)
                    this.loadData()
                    this.updateStatusFilter('avail')

                })
                .catch((e) => {
                    this.error = e;
                    this.message = this.error;
                });
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
