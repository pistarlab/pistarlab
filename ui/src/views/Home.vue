<template>
<div>

    <h1><i class="fa fa-home"></i> Home</h1>
    <b-modal id="modal-create-extension" title="Create New Extension" size="lg" @ok="createNewExtension()">
        <p>
            Create a new extension in your workspace.
        </p>
        <div class="mt-2"></div>
        <label for="newExtensionId">Extension Id:</label>
        <b-form-input id="newExtensionId" v-model="enteredExtensionId" trim></b-form-input> {{newExtensionId}}
        <div class="mt-1"></div>
        <label for="newExtensionName">Extension Name:</label>
        <b-form-input id="newExtensionName" v-model="newExtensionName" trim></b-form-input>
        <label for="newExtensionDescription">Description:</label>
        <b-form-input id="newExtensionDescription" v-model="newExtensionDescription" trim></b-form-input>
    </b-modal>

    <b-modal id="modal-open-workspace" title="Open Workspace Extension" size="lg">
        <div v-if="selectedExtension">
            <h4>Extension Id: {{selectedExtension.id}}</h4>
            <b-alert show>

                NOTE: IDE Integration is under development.
            </b-alert>
            <br />

            <b-button v-if="ideFound" size="sm" @click="openWithIDE(selectedExtension.id)">Open with VS Code</b-button>
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

    <b-container fluid>

        <b-row>

            <b-col cols=5>
                <h3>Recent activity</h3>
                <hr />
                <div class="mt-3"></div>
                <h4 v-b-modal.sessions><i class="fa fa-cubes"></i> Sessions</h4>
                <div>
                    <div v-if="$apollo.queries.sessions.loading">Loading..</div>
                    <div v-else class="">
                        <b-card class="p-0 m-1">
                            <b-table small :items="recentSessions" :fields="sessionFields">
                                <template v-slot:cell(state)="data" class="text-center">
                                    <div class="text-center">
                                        <span v-if="data.item.status =='RUNNING'">
                                            <i class="fas fa-circle" style="color:green"></i>
                                        </span>
                                        <span v-else>
                                            <i class="fas fa-circle"></i>
                                        </span>
                                    </div>
                                </template>
                                <template v-slot:cell(link)="data">

                                    <router-link :to="`/session/view/${data.item.ident}`">{{data.item.ident }} : {{ data.item.envSpecId }}</router-link>

                                </template>

                                <template v-slot:cell(agentId)="data">

                                    <b-link :to="`/agent/view/${data.item.agentId}`"> {{data.item.agentId}}

                                    </b-link>

                                </template>

                                <template v-slot:cell(taskId)="data">
                                    <router-link v-if="data.item.task" :to="`/task/view/${data.item.task.ident}`">{{data.item.task.ident }}</router-link>
                                </template>

                                <template v-slot:cell(created)="data">
                                    {{ timedeltafordate(data.item.created) }} ago
                                </template>

                            </b-table>
                        </b-card>
                        <div class="ml-3 mt-2">
                            <b-link v-b-modal.sessions>View All</b-link>
                        </div>

                    </div>
                </div>
                <div class="mt-3"></div>
                <b-link to="/agent/home">
                    <h4> <i class="fa fa-robot"></i> Agent Instances </h4>
                </b-link>
                <div>
                    <div v-if="$apollo.queries.recentAgents.loading">Loading..</div>
                    <div v-else class="">
                        <b-row>
                            <b-col cols=6 v-for="item in recentAgentsFiltered" v-bind:key="item.ident">
                                <b-link :to="`/agent/view/${item.ident}`">
                                    <b-card class="m-1">

                                        <b-img style="max-height:40px;" :src="`/img/agent_cons/${getImageId(item.ident)}.SVG`" alt="Image" class="rounded-0 svgagent"></b-img>
                                        <span class="ml-2"> {{item.ident}} : {{item.spec.displayedName}}</span>

                                    </b-card>
                                </b-link>
                            </b-col>
                        </b-row>

                    </div>
                </div>
                <div class="ml-3 mt-2">
                    <b-link to="/agent/home">View All</b-link>
                </div>

            </b-col>
            <b-col cols=5>
                <h3>Extensions in your Workspace</h3>
                <hr />
                <b-button class="ml-auto" v-b-modal:modal-create-extension size="sm" variant="success"><i class="fa fa-plus"></i> </b-button>
                <div v-if="workspace" class="mt-3">

                    <b-card v-for="(extension,key) in workspace.extensions" v-bind:key="key" class="mb-0 mt-2">
                        <b-row>
                            <b-col >
                                <div>
                                    <b-link @click="openExtension(extension)">
                                        <h4>{{extension.name}}</h4>
                                    </b-link>
                                </div>
         
                            </b-col>
                            <b-col>
                                <span v-if="extension.status == 'AVAILABLE'">
                                    Not Installed
                                </span>
                                <span v-else>
                                    Installed
                                </span>

                            </b-col>
                            <b-col>
                                <b-button v-if="ideFound" size="sm" @click="openWithIDE(extension.id)" title="View in code editor (VS CODE) - NOTE: Only works when running piStar Lab local"><i class="fas fa-file-code"></i></b-button>
                            </b-col>

                        </b-row>

                    </b-card>

                </div>
                <div class="ml-3 mt-2">
                    <b-link to="/extension/home">View All Extensions</b-link>
                </div>
            </b-col>
            <b-col cols=2 class="text-center">
                <h3>Overview</h3>
                <div v-if="overview">
                    <div class="mb-4">
                        <div class="data_label">Sessions Active</div>
                        <div class="stat_value"> {{overview['active_sessions']}}
                        </div>
                    </div>
                    <div class="mb-4">
                        <div class="data_label">Agent Instances</div>
                        <div class="stat_value"> {{overview['total_agents']}}
                        </div>
                    </div>
                    <div class="mb-4">
                        <div class="data_label">Agent Specs</div>
                        <div class="stat_value"> {{overview['total_agent_specs']}}
                        </div>
                    </div>
                    <div class="mb-4">
                        <div class="data_label">Environment Specs</div>
                        <div class="stat_value"> {{overview['total_env_specs']}}
                        </div>
                    </div>
                    <div class="mb-4">
                        <div class="data_label">Installed Extensions</div>
                        <div class="stat_value"> {{overview['total_installed_extensions']}}
                        </div>
                    </div>

                </div>
            </b-col>

        </b-row>

    </b-container>

    <!-- <b-card title="Missions in Progress">
        <b-card-text>
            <ul>
                <li>Solve Cartpole: Create an Agent which can solve Cartpole-v1 in less than 50k timesteps.</li>
            </ul>

        </b-card-text>
    </b-card>

    <b-card title="My Achievments">
        <b-card-text>
            <div>No achievements yet</div>

        </b-card-text>
    </b-card> -->

    <!-- <div class="mt-4">
    </div> -->

</div>
</template>

<script>
import axios from "axios";

import {
    appConfig
} from "../app.config";
import {
    GET_RECENT_AGENT_SMALL,
    GET_ALL_SESSIONS
} from "../queries";

import {
    getImageId,
    timedeltafordate
} from "../funcs";

const sessionFields = [{
        key: "state",
        label: ""
    },
    {
        key: "link",
        label: "Session"
    },
    {
        key: "agentId",
        label: "Agent",
    },
    {
        key: "created",
        label: "Created",
    }
]
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
        recentAgents: {
            query: GET_RECENT_AGENT_SMALL,
            pollInterval: 10000
        },
        sessions: {
            query: GET_ALL_SESSIONS,
            variables() {
                return {
                    first: 6,
                    last: 0,
                    before: "",
                    after: "",
                    archived: false
                };
            },
            pollInterval: 10000
        },

    },
    data() {
        return {

            sessionFields,
            workspaceFields,
            recentAgents: null,
            sessions: null,
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
            appConfig

        };
    },
    computed: {
        newExtensionId() {
            return "pistarlab-" + this.enteredExtensionId
        },

        recentSessions() {
            if (this.sessions == null) {
                return []

            }
            return this.sessions.edges.map(f => {
                return f.node
            })
        },
        recentAgentsFiltered() {
            if (this.recentAgents == null) {
                return []

            }
            return this.recentAgents.filter((v, i, a) => i < 4)

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
        getImageId,
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
            axios
                .get(`${appConfig.API_URL}/api/workspace/`)
                .then((response) => {
                    this.workspace = response.data.data;
                })
                .catch((error) => {
                    this.message = error;
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
                        console.log(response.data.message)

                    } else {
                        console.log(response.data.message)
                    }

                })
                .catch((error) => {
                    this.message = error;
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
