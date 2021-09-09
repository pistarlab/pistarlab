<template>
<div class="page">
    <div class="page-content">
        <h1><i class="fa fa-puzzle-piece"></i> Extensions</h1>
        <div class="mt-4"></div>

        <b-modal id="modal-restart" title="Restart" size="lg">
            <RestartDialog></RestartDialog>
        </b-modal>
        <b-modal id="modal-logviewer" title="Extension Manager Logs" size="xl">
            <LogViewer :nocard="true" :logStreamUrl="`${appConfig.API_URL}/api/stream/scoped/extension_manager`"> </LogViewer>
        </b-modal>
        <div v-if="!manageExtensionId">
            <b-navbar>

                Filter by Extension Status:
                <b-button-toolbar>
                    <b-button-group size="sm">
                        <b-button class="mr-0" :pressed="selectedStatus==''" @click="updateStatusFilter('')">Any</b-button>
                        <b-button class="mr-0" :pressed="selectedStatus=='installed'" @click="updateStatusFilter('installed')">Installed</b-button>
                        <b-button class="mr-0" :pressed="selectedStatus=='avail'" @click="updateStatusFilter('avail')">Not Installed</b-button>

                    </b-button-group>
                </b-button-toolbar>
                <b-form-checkbox class="ml-2" switch v-model="onlyWorkspaceExtensions">Show only Workspace Extensions</b-form-checkbox>

                <!-- <span class="mr-2 ml-5">Category Filter:</span> -->
                <!-- <b-button-group size="sm">
            <b-button class="mr-2" v-for="(btn, idx) in filterCategories" :key="idx" :pressed.sync="btn.state"  @click="updateList()" variant="info" pill>{{ btn.caption }} </b-button>
   
        </b-button-group> -->
                <b-button class="ml-auto mr-2" v-b-modal:modal-logviewer size="sm" variant="info"><i class="fa fa-bug"></i> Extension Logs</b-button>

            </b-navbar>
            <b-navbar toggleable="lg" type="light" variant="alert">

            </b-navbar>
            <b-form-input v-model="searchtext" placeholder="Search Extensions" style="width:250px;" class='ml-2'></b-form-input>

        </div>
        <div class="mt-4"></div>
        <div v-if="!loading">

            <div v-if="Object.keys(filteredExtensions).length >0">
                <b-container fluid>
                    <div v-for="(item, idx) in filteredExtensions" :key="idx">
                        <b-row>
                            <b-col>
                                <div v-b-toggle="'collapse_'+idx">
                                    <span class="hover h4"><i class="fa fa-puzzle-piece"> </i> {{item.name}}</span>       <b-link v-b-popover.hover.top="'This extension is in your workspace.'" class="ml-2" to="/workspace/home">
                                    <b-badge v-if="item.source.name == 'Workspace'" pill variant="warning" class="mr-2"><i class="fa fa-code"></i> Workspace</b-badge>
                                </b-link>
                                    <p class="desc">{{item.description}}</p>
                                </div>
                            </b-col>
                 
                            <b-col>
                                <div class="text-right">
                                    <b-button v-if="item.status == 'AVAILABLE'" size="sm" variant="info" @click="installExtension(item.id,item.version);">Install</b-button>
                                    <b-button v-else-if="item.status == 'INSTALLING'" size="sm" variant="" disabled>
                                        <b-spinner small type="grow"></b-spinner>Installing...
                                    </b-button>
                                    <div v-else>
                                        <b-button v-if="item.status == 'INSTALL_FAILED'" size="sm" variant="secondary" class="mr-2" @click="installExtension(item.id,item.version);">Retry Install</b-button>
                                        <b-button v-if="item.status == 'INSTALLED'" size="sm" variant="secondary" class="mr-2" @click="installExtension(item.id,item.version);">Reinstall</b-button>

                                        <b-button class="mr-2" size="sm" variant="" @click="removeExtension(item.id,item.version)">Uninstall</b-button>
                                        <!-- <b-button v-if="['INSTALLED','PREPPED_RELOAD'].includes(item.status)" class="mr-2" size="sm" variant="secondary" @click="reloadExtension(item.id,item.version)">Reload</b-button> -->
                                    </div>
                                </div>
                            </b-col>
                        </b-row>

                        <b-row class="small ">
                            <b-col>

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
                                <div>
                                    <span class="data_label mt-1">Original Author: </span>
                                    <span>{{item.original_author}}</span>
                                </div>
                                <div>
                                    <span class="data_label mt-1">Extension Author: </span>
                                    <span>{{item.extension_author}}</span>
                                </div>
                            </b-col>
                            <b-col class="">
                                <div>
                                    <span class="data_label mt-1">Source Name: </span>
                                    <span>{{item.source.name}} </span>

                                </div>
                                <div v-if="item.source.type">
                                    <span class="data_label mt-1">Source Type: </span>

                                    <span>{{item.source.type}}</span>
                                </div>
                                <div v-if="item.source.path">
                                    <span class="data_label mt-1">Source Path: </span>

                                    <span>{{item.source.path}}</span>
                                </div>
                                <div v-if="item.full_path">
                                    <span class="data_label mt-1">Full Path: </span>

                                    <span style="color:red">{{item.full_path}}</span>
                                </div>
                                <div v-if="item.installation_instructions">
                                    <span class="data_label mt-1">Additional Installation Instructions: </span>

                                    <span style="color:red">{{item.installation_instructions}}</span>
                                </div>
                            </b-col>
                            <b-col class="">
                                <span class="data_label mt-1">State: </span>
                                <span v-if="item.status == 'PREPPED_RELOAD'" style="color:yellow">**Restart piSTAR Lab to complete installation* <b-button v-b-modal:modal-restart size="sm">Restart Now</b-button></span>
                                <span v-else>{{item.status}} </span>
                                <span v-if="(item.status == 'INSTALL_FAILED' || item.status == 'UNINSTALL_FAILED') && item.status_msg">
                                    <pre>{{item.status_msg}}</pre>
                                </span>
                            </b-col>
                        </b-row>

                        <span v-if="idx != Object.keys(filteredExtensions).length - 1">
                            <hr /> </span>

                    </div>
                </b-container>
            </div>

            <div v-else>

                No Extensions Found

            </div>
        </div>
        <div v-else>
            loading...
        </div>
    </div>
    <HelpInfo contentId="extensions"></HelpInfo>

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
        label: "Extension UID",
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
import RestartDialog from "../components/RestartDialog.vue"

export default {
    name: "ExtensionHome",
    components: {
        LogViewer,RestartDialog
    },
    props: {
        showWorkspaceExtensions: Boolean,
        manageExtensionId: String,
        category:{
            type:String,
            default:null
        }
    },
    data() {
        return {
            appConfig,
            searchtext: "",
            fields: fields,
            allExtensions: {},
            error: "",
            selected: [],
            loading: true,

            selectedStatus: "",
            onlyWorkspaceExtensions: false,
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
        filteredExtensions() {
            let extensionlist = {}
            if (this.manageExtensionId) {
                extensionlist = this.filtered.filter(extension => {
                    return extension.id == this.manageExtensionId;
                })
            } else {

                if (this.selectedStatus == 'installed') {
                    extensionlist = this.filtered.filter(extension => {
                        return extension.status != "AVAILABLE";
                    })

                } else if (this.selectedStatus == 'avail') {
                    extensionlist = this.filtered.filter(extension => {
                        return extension.status == "AVAILABLE" || extension.status == "INSTALLING" || extension.status == "INSTALL_FAILED" || extension.status == "UNINSTALL_FAILED";
                    })
                } else {
                    extensionlist = this.filtered
                }
            }

            return extensionlist.sort((a, b) => {
                return a.name > b.name
            })
        },

        filtered() {
            let categories = new Set();

            Object.keys(this.filterCategories).forEach(cat => {
                if (this.filterCategories[cat]["state"]) {
                    categories.add(cat);
                }
            });

            const filteredList = []
            Object.values(this.allExtensions).forEach((extension) => {
                let include = false;

                if (this.onlyWorkspaceExtensions && extension['source']['name'] != "Workspace") {
                    return
                }
                for (const cat of extension.categories) {
                    if (categories.has(cat)) {
                        include = true
                        break;
                    }
                }
                if (extension.categories.length == 0 || (extension.categories.length == this.filterCategories.length) || include) {

                    filteredList.push(extension)
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
        getExtensionKey(extensionId, extensionVersion) {
            return `${extensionId}__v${extensionVersion}`
        },

        updateList(btn) {
            this.loadData();

        },
        installExtension(extensionId, extensionVersion) {
            console.log("Installing " + this.getExtensionKey(extensionId, extensionVersion));
            const extension = this.allExtensions[this.getExtensionKey(extensionId, extensionVersion)]
            extension.status = "INSTALLING"
            this.$set(this.allExtensions, this.getExtensionKey(extensionId, extensionVersion), extension)

            axios
                .get(`${appConfig.API_URL}/api/extensions/action/install/${extensionId}/${extensionVersion}`)
                .then((response) => {
                    this.loadData();
                    this.updateStatusFilter('installed')
                })
                .catch(function (error) {
                    this.errorMessage = error;
                    this.loadData();
                });

        },
        removeExtension(extensionId, extensionVersion) {
            console.log("Uninstall " + this.getExtensionKey(extensionId, extensionVersion));
            this.allExtensions[this.getExtensionKey(extensionId, extensionVersion)].status = "REMOVING"

            axios
                .get(`${appConfig.API_URL}/api/extensions/action/uninstall/${extensionId}/${extensionVersion}`)
                .then((response) => {
                    this.errorMessage = JSON.stringify(response.data);
                    this.loadData();
                })
                .catch(function (error) {
                    this.errorMessage = error;
                });

            this.loadData();
        },
        reloadExtension(extensionId, extensionVersion) {
            console.log("Reloading " + this.getExtensionKey(extensionId, extensionVersion));

            axios
                .get(`${appConfig.API_URL}/api/extensions/action/reload/${extensionId}/${extensionVersion}`)
                .then((response) => {
                    this.errorMessage = JSON.stringify(response.data);
                    this.loadData();
                })
                .catch(function (error) {
                    this.errorMessage = error;
                });
            this.loadData();

        },
        openLink(url) {
            console.log(url)
            this.$router.push({
                path: url
            }).catch((x) => {
                //
            });
            this.$emit("hide")

        },
        loadData() {
            this.loading = true

            axios
                .get(`${appConfig.API_URL}/api/extensions/list`)
                .then((response) => {
                    this.allExtensions = response.data["items"]
                    this.loading = false

                })
                .catch((e) => {
                    this.error = e;
                    this.loading = false
                });
        },
        updateStatusFilter(status) {
            console.log(status)
            this.selectedStatus = status
        }
    },
    // Fetches posts when the component is created.
    created() {
        if (this.showWorkspaceExtensions) {
            this.onlyWorkspaceExtensions = true
        }
        this.selectedStatus = ""
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
