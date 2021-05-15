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

    <b-modal id="modal-open-workspace" title="Open Workspace">
        Currently under development. For now, navigate to path below in your IDE of choice.
        <div v-if="workspace"> {{workspace.path}}
        </div>

    </b-modal>
    <b-modal id="plugin-manager-workspace" size="xl" title="Plugins" scrollable :hide-footer="true">
        <PluginManager :showWorkspacePlugins="true"></PluginManager>
        <div class="mb-5"></div>

    </b-modal>
    <b-container fluid>

        <b-row>

            <b-col cols=6>
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
                                <template v-slot:cell(agentId)="data">

                                    <b-link :to="`/agent/view/${data.item.agentId}`"> {{data.item.agentId}}

                                    </b-link>

                                </template>
                                <template v-slot:cell(link)="data">

                                    <router-link :to="`/session/view/${data.item.ident}`">{{data.item.ident }}</router-link>

                                </template>

                                <template v-slot:cell(taskId)="data">
                                    <router-link v-if="data.item.task" :to="`/task/view/${data.item.task.ident}`">{{data.item.task.ident }}</router-link>
                                </template>
                                <template v-slot:cell(envSpecId)="data">
                                    <router-link :to="`/env_spec/view/${data.item.envSpecId}`">{{ data.item.envSpecId }}</router-link>
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
                    <h4> <i class="fa fa-robot"></i> Agents </h4>
                </b-link>
                <div>
                    <div v-if="$apollo.queries.recentAgents.loading">Loading..</div>
                    <div v-else class="">
                        <b-row>
                            <b-col cols=6 v-for="item in recentAgentsFiltered" v-bind:key="item.ident">
                                <b-link :to="`/agent/view/${item.ident}`">
                                    <b-card class="m-1">

                                        <b-img style="max-height:40px;" :src="`/img/agent_cons/${getImageId(item.ident)}.SVG`" alt="Image" class="rounded-0 svgagent"></b-img>
                                        <span class="ml-2"> {{item.ident}} {{item.spec.displayedName}}</span>

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
            <b-col cols=6>
                <h3>Workspace Plugins</h3>
                <hr />
                <b-button class="ml-auto" v-b-modal:modal-create-plugin size="sm" variant="success"><i class="fa fa-plus"></i> </b-button>
                <b-button class="ml-2" v-b-modal:modal-open-workspace size="sm" variant="secondary"><i class="fas fa-folder"></i> </b-button>
                <div v-if="workspace" class="mt-3">

                    <b-card v-for="(plugin,key) in workspace.plugins" v-bind:key="key" class="mb-0 mt-2">
                        <b-row>
                            <b-col cols=6>
                                <div>
                                    <h4>{{plugin.name}}</h4>
                                </div>
                                <div>{{plugin.id}}</div>
                                <div></div>

                            </b-col>
                            <b-col cols=2>
                                <span v-if="plugin.status == 'AVAILABLE'">
                                    Not Installed
                                </span>
                                <span v-else>
                                    Installed
                                </span>

                            </b-col>
                            <b-col class="text-center">
                                <b-button-group size="sm">
                                    <b-button v-b-modal.plugin-manager-workspace>Manage</b-button>
                                </b-button-group>

                            </b-col>
                        </b-row>

                    </b-card>

                </div>
                <div class="ml-3 mt-2">
                    <b-link v-b-modal.plugin-manager-workspace>View All Plugins</b-link>
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
        label: "ID"
    },
    {
        key: "envSpecId",
        label: "Environment Spec"
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

import PluginManager from "./PluginHome.vue";

export default {
    name: "Home",
    components: {
        PluginManager
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
            projectName: "default",
            packageName: "",
            message: ".",
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

    },

    created() {
        this.loadWorkspace()
        //
    },
};
</script>

<style >
</style>
