<template>
<div>

    <h1><i class="fa fa-home"></i> Home</h1>
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

    <b-modal id="modal-open-workspace" title="Open Workspace Plugin" size="lg">
        <div v-if="selectedPlugin">
            <h4>Plugin Id: {{selectedPlugin.id}}</h4>
            <b-alert show>

                NOTE: IDE Integration is under development.
            </b-alert>
            <div>
                <b-button size="sm" :to="`/plugin/home/?managePluginId=${selectedPlugin.id}`">Manage</b-button>
            </div>
            <div class="mt-3">
            Open IDE or File browser of choice to path below:
            <pre v-if="selectedPlugin">{{selectedPlugin.full_path}}</pre>
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
                <h3>Workspace Plugins</h3>
                <hr />
                <b-button class="ml-auto" v-b-modal:modal-create-plugin size="sm" variant="success"><i class="fa fa-plus"></i> </b-button>
                <div v-if="workspace" class="mt-3">

                    <b-card v-for="(plugin,key) in workspace.plugins" v-bind:key="key" class="mb-0 mt-2">
                        <b-row>
                            <b-col cols=6>
                                <div>
                                    <b-link @click="openPlugin(plugin)">
                                        <h4>{{plugin.name}}</h4>
                                    </b-link>
                                </div>
                                <div>{{plugin.id}}</div>
                                <div></div>

                            </b-col>
                            <b-col cols=4>
                                <span v-if="plugin.status == 'AVAILABLE'">
                                    Not Installed
                                </span>
                                <span v-else>
                                    Installed
                                </span>

                            </b-col>
                            <b-col cols=2>
                                <b-button size="sm" @click="openPlugin(plugin)">Open</b-button>
                            </b-col>

                        </b-row>

                    </b-card>

                </div>
                <div class="ml-3 mt-2">
                    <b-link to="/plugin/home">View All Plugins</b-link>
                </div>
            </b-col>
            <b-col cols=2 class="text-center">
                <h3>Overview</h3>
                <div v-if="overview">
                    <div class="mb-4">
                        <div class="data_label">Sessions</div>
                        <div class="stat_value"> {{overview['total_sessions']}}
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
                        <div class="data_label">Installed Plugins</div>
                        <div class="stat_value"> {{overview['total_installed_plugins']}}
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
            enteredPluginId: "",
            newPluginName: "",
            newPluginDescription: "",
            selectedPlugin: null,
            projectName: "default",
            packageName: "",
            message: ".",
            overview: null,
            appConfig,
        };
    },
    computed: {
        newPluginId() {
            return "pistarlab-" + this.enteredPluginId
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
        openPlugin(plugin) {
            this.selectedPlugin = plugin

            this.$bvModal.show("modal-open-workspace")

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

    },

    created() {
        this.loadWorkspace()
        this.loadOverview()
        //
    },
};
</script>

<style >
</style>
