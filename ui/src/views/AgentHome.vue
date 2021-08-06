<template>
<div>
    <h1><i class="fa fa-robot"></i> Agents</h1>

    <div class="mt-4"></div>

    <b-button-toolbar>

        <b-modal id="agentnew" size="xl" :hide-header="true" :hide-footer="true">
            <AgentNew :specId="selectedSpecId" @agentCreated="agentCreated($event)"></AgentNew>
        </b-modal>
        <b-modal id="agentspecs" size="xl" :hide-header="true" :hide-footer="true">
            <AgentSpecs @specSelected="newAgentModal($event)"></AgentSpecs>
        </b-modal>
        <b-button variant="primary" size="sm" v-b-modal:agentspecs>New</b-button>
        <b-button class="ml-2" size="sm" v-b-toggle="`sess_group`">Toggle Sessions</b-button>
    </b-button-toolbar>
    <div class="mt-4"></div>

    <b-container fluid>

        <b-row>
            <b-col>
                <h3>Recently Active</h3>
            </b-col>
        </b-row>
        <div class="mt-2"></div>
        <div v-if="$apollo.queries.recentAgents.loading">Loading..</div>
        <div v-else>
            <div v-if="items.length > 0">
                <b-row>
                    <b-col class="d-flex flex-wrap mb-4">
                        <span v-for="item in items" v-bind:key="item.ident" class="m-3">
                            <b-card no-body header-bg-variant="info" header-text-variant="white" class="h-100 card-shadow card-flyer" style="width: 260px">
                                <template v-slot:header>
                                    <div class="custom-card-header mb-2">
                                        <b-link style="color: white" :to="`/agent/view/${item.ident}`">{{ item.ident }}</b-link>
                                    </div>

                                </template>
                                <b-card-text class="h-100">
                                    <b-container class="mt-2">
                                        <b-row>
                                            <b-col>
                                                <b-link :to="`/agent/view/${item.ident}`">
                                                    <b-card-img height="160px" :src="`/img/agent_cons/${getImageId(item.ident)}.SVG`" alt="Image" class="rounded-0 svgagent"></b-card-img>
                                                </b-link>
                                                <div class="mt-2">
                                                    <b-badge pill v-for="(tag,id) in item.tags.edges" v-bind:key="id" variant="tag" class="mr-1">{{tag.node.tagId}}</b-badge>
                                                    <div v-if="item.tags.edges.length==0" style="height:1.5em"></div>

                                                </div>
                                            </b-col>
                                        </b-row>
                                        <hr class="mt-1 mb-1" />
                                        <b-row>
                                            <b-col>

                                                <div class="mt-1">

                                                    <span class="data_label mr-1 mt-0">Spec:</span>
                                                    <span class="">
                                                        <b-link :to="`/agent_spec/${item.specId}`">{{ item.specId }}</b-link>
                                                    </span>
                                                </div>

                                                <div class="mt-1">
                                                    <span class="data_label mr-1 mt-0">Created </span>
                                                    <span class=""> {{ timedeltafordate(item.created) }} ago</span>
                                                </div>
                                               
                                                    <b-collapse id="sess_group" class="mt-2">
                                                        <div class="mt-1">
                                                        <div class="ml-0 pl-0 " v-if="item.recentSessions.length ==0">None</div>
                                                        <div class="ml-2 pl-0" v-for="(session, widx) in item.recentSessions" v-bind:key="widx">
                                                            <i v-if="session.status=='RUNNING'" class="fa fa-circle" style="color:green"></i>
                                                            <i v-else class="fa fa-circle"></i>

                                                            <b-link class="ml-2" :to="`/session/view/${session.ident}`">{{ session.envSpecId }} ({{session.ident}}) <span v-if="session.sessionType == 'RL_MULTIPLAYER_SINGLEAGENT_SESS'"> <i class="fas fa-cubes" title="multiagent"></i> </span>
                                                            </b-link>
                                                            <span class="ml-4">
                                                                <b-link v-if="session.parentSessionId" class="" :to="`/session/view/${session.parentSessionId}`" title="parent session"><i class="fas fa-cubes" title="multiagent"></i> {{session.parentSessionId}}</b-link>
                                                            </span>
                                                        </div>

                                                </div>
                                                    </b-collapse>

                                            </b-col>
                                        </b-row>

                                    </b-container>
                                </b-card-text>
                                <template v-slot:footer>

                                    <b-button-toolbar class="mr-auto">
                                        <b-button title="Assign Task" variant="dark" class="mr-1" size="sm" :to="`/task/new/agenttask/?agentUid=${item.ident}`"><i class="fa fa-plus"></i></b-button>
                                                                                            


                                    </b-button-toolbar>

                                </template>
                            </b-card>
                        </span>
                    </b-col>
                </b-row>
                <b-row>
                    <b-col>

                        <p>{{ error }}</p>
                    </b-col>
                </b-row>
            </div>
            <div v-else>

                <b-row>
                    <b-col>
                        No Active Agents Available
                        <div class="mt-4">
                        </div>
                    </b-col>
                </b-row>
            </div>

        </div>
        <b-row>
            <b-col>

                <div>
                    <b-link :to="`/agent/instances`">View All Agent Instances</b-link>
                </div>
            </b-col>
        </b-row>
    </b-container>
</div>
</template>

<script>
// @ is an alias to /src
import axios from "axios";
import gql from "graphql-tag";
import {
    appConfig
} from "../app.config"

import {
    timedeltafordate,
    timepretty,
    timelength,
    timedelta
} from "../funcs";

import {
    GET_RECENT_AGENTS
} from "../queries";

import AgentSpecs from "../components/AgentSpecs.vue";
import AgentNew from "../components/AgentNew.vue";

export default {
    name: "Agents",
    components: {
        AgentSpecs,
        AgentNew

    },
    apollo: {
        recentAgents: {
            query: GET_RECENT_AGENTS,
            pollInterval: 2000
        },
    },
    data() {
        return {
            recentAgents: [],
            selectedSpecId: null,
            searchQuery: "",
            itemsPerRow: 3,
            n: 13,
            error: "",
            taskEntities: [],
        };
    },
    computed: {
        items() {
            const agents = [];
            this.recentAgents.forEach((agentNode) => {
                const agent = {
                    ...agentNode
                };
                agents.push(agent);
            });
            return agents;
        },
    },
    methods: {

        timedeltafordate,
        timedelta,
        timelength,
        agentCreated(agentId) {
            this.$router.push({
                path: `/agent/view/${agentId}`,
            });

        },
        newAgentModal(specId) {
            this.selectedSpecId = specId
            this.$bvModal.hide("agentspecs")
            this.$bvModal.show("agentnew")
        },

        shutdownAgent(uid) {
            axios
                .get(`${appConfig.API_URL}/api/agent_control/SHUTDOWN?uid=${uid}`)
                .then((response) => {
                    // JSON responses are automatically parsed.
                    this.message = response.data["message"];
                    this.$apollo.queries.recentAgents.refetch();
                })
                .catch((e) => {
                    this.error = e;
                    this.message = this.error;
                });
            console.log(uid);
        },
        getImageId(uid) {
            let id = parseInt(uid.split("-")[1]);
            return id % 19;
        },
    },

    created() {
        // this.$store.dispatch("fetchAgents");
    },
};
</script>
