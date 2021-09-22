<template>
<div class="page">
    <div class="page-content">

        <h1><i class="fa fa-home"></i> Workspace</h1>
        <b-modal id="modal-create-extension" title="Create New Extension" size="lg" @ok="createNewExtension()">
            <p>
                Create a new extension in your workspace.  
                <br/>
                
            </p>
            <!-- <b-alert show>
                Agents and Environments created in Workspace Extension cannot be published to the community hub.
            </b-alert> -->
            <div class="mt-2"></div>
            <label for="newExtensionId">Extension Id:</label>
            <b-form-input id="newExtensionId" v-model="enteredExtensionId" trim></b-form-input> {{newExtensionId}}
            <div class="mt-1"></div>
            <label for="newExtensionName">Extension Name:</label>
            <b-form-input id="newExtensionName" v-model="newExtensionName" trim></b-form-input>
            <label for="newExtensionDescription">Description:</label>
            <b-form-input id="newExtensionDescription" v-model="newExtensionDescription" trim></b-form-input>
        </b-modal>

        <b-modal id="modal-open-workspace" title="Open Extension" size="lg">
            <div v-if="selectedExtension">
                <h4>Extension Id: {{selectedExtension.id}}</h4>


                <b-alert show>

                    NOTE: IDE Integration is under development.
                </b-alert>
                <br />

                <b-button class="mr-2" v-if="ideFound" size="sm" @click="openWithIDE(selectedExtension.id)">Open with VS Code</b-button>
                <div v-else>VSCode not nound. See https://code.visualstudio.com/</div>
                <br />
                <br />
                <div>
                    <b-link size="sm" :to="`/extension/home/?manageExtensionId=${selectedExtension.id}`">Manage Extension</b-link>

                </div>
                <div class="mt-3">
                    Open IDE or File browser of choice to path below:
                    <pre v-if="selectedExtension">{{selectedExtension.full_path}}</pre>
                </div>
            </div>
        </b-modal>


        <b-container fluid v-if="!this.loading && !this.loadingIDE"  >

            <b-row>

                <b-col>
                    <h3>Your Extensions</h3>
                    <hr />

                    <b-button class="ml-auto" v-b-modal:modal-create-extension size="sm" variant="success"><i class="fa fa-plus"></i> Start New Extension Project</b-button>
                    <div class="mt-4 mb-4">
                        <div v-if=" this.ideFound">
                            IDE Status: VS Code seems to be installed.
                        </div>
                        <div v-else>
                            IDE Status: VSCode not found. For extension development, we recommend installing an IDE such as <b-link target="_blank" href="https://code.visualstudio.com/">VS Code</b-link>.
                        </div>
                    </div>
                    <div class="mt-3">

                        <div v-if="workspace && workspace.extensions && workspace.extensions.length>0">
                            <b-card v-for="(extension,key) in workspace.extensions" v-bind:key="key" class="mb-0 mt-2">
                                <b-row>
                                    <b-col>
                                        <div>
                                            <b-link @click="openExtension(extension)">
                                                <h4><i class="fa fa-puzzle-piece"></i> {{extension.name}}</h4>
                                            </b-link>
                                        </div>

                                    </b-col>
                                    <b-col class="text-center">
                                        <span v-if="extension.status == 'AVAILABLE'">
                                            Not Installed
                                        </span>
                                        <span v-else>
                                            Installed
                                        </span>

                                    </b-col>
                                    <b-col class="text-right">
                                        <b-button v-b-popover.hover.top="'View in code editor (VS CODE) - NOTE: Only works when running piStar Lab local'" class="mr-2" v-if="ideFound" size="sm" @click="openWithIDE(extension.id)"><i class="fas fa-file-code"></i> Open in VS Code</b-button>
                                        <b-button v-b-popover.hover.top="'Install/Uninstall'" size="sm" :to="`/extension/home/?manageExtensionId=${extension.id}`"><i class="fa fa-redo-alt"></i> Manage</b-button>
                                    </b-col>

                                </b-row>
                                <b-row>
                                    <b-col>
                                        <div class="ml-4">
                                            <div class="small">
                                                Id: {{extension.id}}
                                            </div>
                                            <div class="small">
                                                Path: {{extension.full_path}}
                                            </div>
                                        </div>
                                    </b-col>
                                </b-row>

                            </b-card>
                        </div>
                        <div v-else>
                            
                            <i class="fa fa-info-circle mr-1"></i> No extensions found in your workspace.
                        </div>

                    </div>
                    <!-- <div v-else>
                        loading...
                    </div> -->
                    <div class="ml-3 mt-4">
                        <b-link to="/extension/home">View All Extensions</b-link>
                    </div>
                    
                </b-col>

            </b-row>

        </b-container>

    </div>
    <HelpInfo contentId="workspace"></HelpInfo>
</div>
</template>

<script>
import axios from "axios";

import {
    appConfig
} from "../app.config";

import {
    timedeltafordate
} from "../funcs";

const workspaceFields = [{
        key: "name",
        label: "Name"
    },
    {
        key: "actions",
        label: ""
    }
]

export default {
    name: "Home",
    components: {
        //
    },
    apollo: {
        //

    },
    data() {
        return {

            workspaceFields,
            recentAgents: null,
            workspace: null,
            enteredExtensionId: "",
            newExtensionName: "",
            newExtensionDescription: "",
            selectedExtension: null,
            projectName: "default",
            packageName: "",
            message: ".",
            overview: null,
            ideFound: false,
            loading: true,
            loadingIDE:true,
            appConfig

        };
    },
    computed: {
        newExtensionId() {
            return "pistarlab-" + this.enteredExtensionId
        },

        recentEnvs() {
            if (this.sessions == null) {
                return []

            }
            return this.sessions.edges.map(f => {
                return f.node.envSpecId

            }).filter((v, i, a) => a.indexOf(v) === i)
        }

    },
    methods: {
        timedeltafordate,
        openExtension(extension) {
            this.selectedExtension = extension

            this.$bvModal.show("modal-open-workspace")

        },
        createNewExtension() {

            let outgoingData = {
                'extension_id': this.newExtensionId,
                'extension_name': this.newExtensionName,
                'description': this.newExtensionDescription

            }
            axios
                .post(`${appConfig.API_URL}/api/extension/create`, outgoingData)
                .then((response) => {
                    console.log(response)
                    this.loadWorkspace()

                })
                .catch((e) => {
                    this.error = e;
                    this.message = this.error;
                });
        },
        loadWorkspace() {
            this.loading = true
            axios
                .get(`${appConfig.API_URL}/api/workspace/`)
                .then((response) => {
                    this.workspace = response.data.data;
                    this.loading = false
                })
                .catch((error) => {
                    this.message = error;
                    this.loading = false
                });
        },
        loadOverview() {
            axios
                .get(`${appConfig.API_URL}/api/overview/`)
                .then((response) => {
                    this.overview = response.data;
                })
                .catch((error) => {
                    this.message = error;
                });
        },
        checkForIDE() {
            console.log("Request for opening extension in IDE")
            axios
                .get(`${appConfig.API_URL}/api/check_for_ide/`)
                .then((response) => {
                    if (response.data.success) {
                        this.ideFound = true;
                        this.loadingIDE = false;
                        console.log(response.data.message)

                    } else {
                        console.log(response.data.message)
                        this.loadingIDE = false;
                    }

                })
                .catch((error) => {
                    this.message = error;
                    this.loadingIDE = false;
                });
        },
        openWithIDE(extensionId) {
            console.log("Request for opening extension in IDE")
            axios
                .get(`${appConfig.API_URL}/api/open_extension_with_ide/${extensionId}`)
                .then((response) => {
                    this.overview = response.data;
                })
                .catch((error) => {
                    this.message = error;
                });
        },

    },

    created() {
        this.loadWorkspace()
        this.loadOverview()
        this.checkForIDE()
        //
    },
};
</script>

<style >
</style>
