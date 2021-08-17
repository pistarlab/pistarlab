<template>
<div v-if="item">

    <h1><i class="fa fa-gamepad"></i> Environment Spec: {{item.ident}}</h1>
    <div class="mt-4"></div>

    <b-modal id="meta-modal">
        <pre v-if="item && item.meta">{{ JSON.parse(item.meta) }}</pre>
    </b-modal>

    <b-modal id="config-modal">
        <pre v-if="item && item.meta">{{ JSON.parse(item.config) }}</pre>
    </b-modal>
    <b-button-group size="sm">
        <b-button variant="primary" :to="`/task/new/agenttask/?envSpecId=${item.ident}`" size="sm">Assign</b-button>

        <b-button variant="secondary" v-b-modal="'meta-modal'" class="ml-1" size="sm">Metadata</b-button>
        <b-button variant="secondary" v-b-modal="'config-modal'" class="ml-1" size="sm">Config</b-button>
    </b-button-group>
    <div class="mt-4"></div>
    <b-alert v-if="item.environment.disabled" show variant="warning">
        Warning, this environment has been <b>disabled</b>.  This is happens when the Environment's extension is uninstalled.
    </b-alert>
    <b-card>
        <b-container fluid>
            <b-row>
                <b-col>

                    <div>
                        <span class="stat_label">Spec Id: </span> {{item.ident}}
                    </div>
                    <div>
                        <span class="stat_label">Entry Point: </span>{{item.entryPoint}}
                    </div>
                    <div>
                        <span class="stat_label">Environment Id: </span>{{item.environment.ident}}
                    </div>
                    <div>
                        <span class="stat_label">Extension Id: </span> {{item.environment.extensionId}}
                    </div>
                                        <div>
                        <span class="stat_label">Disabled: </span> {{item.environment.disabled}}
                    </div>
                </b-col>
                <b-col>
                    <div class="text-center image-box">
                        <img :src="`${appConfig.API_URL}/api/env_preview_image/${item.ident}`" alt="" style="max-height:200px;" />
                    </div>
                </b-col>
            </b-row>
            <h3>Sessions Ranked by Mean Episode Reward</h3>
            <p class="desc">
                Multi Agent Parent sessions not included below.
            </p>
            <!-- <b-alert show variant="warning">Partially Broken: Only works when using PostgreSQL database (TODO: fix me)</b-alert> -->
            <div class="mt-2"></div>
            <b-row>
                <b-col>
                    <b-table show-empty empty-text="No Sessions Found" hover table-busy :items="orderedSessions" :fields="sessionFields" :dark="false" :small="false" :borderless="false">
                        <template v-slot:cell(ident)="data">
                            <!-- `data.value` is the value after formatted by the Formatter -->
                            <router-link :to="`/session/view/${data.item.ident}`">{{data.item.ident}}</router-link>
                        </template>
                        <template v-slot:cell(agent)="data">
                            <!-- `data.value` is the value after formatted by the Formatter -->

                            <router-link v-if="data.item.agent" :to="`/agent/view/${data.item.agent.ident}`">{{data.item.agent.ident}}: {{data.item.agent.specId}}</router-link>
                        </template>
                                                <template v-slot:cell(summary)="data">
                            <!-- `data.value` is the value after formatted by the Formatter -->

                            {{data.item.summary}}
                        </template>
                    </b-table>

                </b-col>

            </b-row>

        </b-container>
    </b-card>
</div>
</template>

<script>
// @ is an alias to /src
import axios from "axios";
import {
    appConfig
} from "../app.config";
import gql from "graphql-tag";

const fields = [{
        key: "info.uid",
        label: ""
    },
    {
        key: "info.name",
        label: "Label",
        sortable: true
    }
];

const sessionFields = [{
        key: "ident",
        label: "Session"
    },
    {
        key: "agent",
        label: "Agent",
        sortable: true
    },
    {
        key: "summary.best_ep_reward_mean_windowed",
        label: "Best Episode Reward Mean (Windowed)",
        sortable: true
    },
            {
        key: "summary.step_count",
        label: "Step Count",
        sortable: true
    },
        {
        key: "summary.episode_count",
        label: "Episode Count",
        sortable: true
    },
            {
        key: "summary.mean_reward_per_episode",
        label: "Mean Reward per episode",
        sortable: true
    }
];

const GET_ENV_SPEC = gql `
 query GetEnvSpec($ident: String!) {
    item:envSpec(ident: $ident) {
        id
      ident
      entryPoint
      config
      created
      environment {
          id
          ident
                extensionId
                disabled

      }
      meta
      }
    
  }
`;

const GET_BEST_SESSIONS = gql `
    query bestSessionsForEnvSpec($envSpecId: String!, $statName: String!) {
    bestSessions:bestSessionsForEnvSpec(envSpecId: $envSpecId, statName: $statName) {
        id
        ident
        sessionType
        parentSessionId
        agent {
            id
            ident
            specId
        }
        summary 
    }
    }
`;

export default {
    name: "env",
    components: {},
    apollo: {
        item: {
            query: GET_ENV_SPEC,
            variables() {
                return {
                    ident: this.specId,
                };
            },

        },
        bestSessions: {
            query: GET_BEST_SESSIONS,
            variables() {
                return {
                    envSpecId: this.specId,
                    statName: 'best_ep_reward_mean_windowed', //TODO: not in use
                };
            },

        },
    },
    data() {
        return {
            appConfig,
            sessionFields,
            item: null,
            bestSessions: [],
            error: ""
        };
    },
    props: {
        specId: String
    },
    computed: {

        orderedSessions() {
            if (this.bestSessions == null || this.bestSessions.length == 0) {
                return []
            } else {
                return this.bestSessions.slice().filter( a=> a.sessionType == "RL_SINGLEPLAYER_SESS").sort((a, b) =>
                    b.summary.best_ep_reward_mean_windowed - a.summary.best_ep_reward_mean_windowed
                )
                
            }

        },
        config() {
            if (this.item)
                return JSON.parse(this.item.config)
            else
                return ""

        },
        meta() {
            if (this.item)
                return JSON.parse(this.item.meta)
            else return ""
        }

    },
    created() {
        //
    }
};
</script>

<style>

</style>
