<template>
<div class="page">
    <div class="page-content">
        <h1><i class="fa fa-gamepad"></i> Environments</h1>
        <div class="mt-4"></div>

        <b-modal id="modal-addcustom">
            TODO
        </b-modal>

        <div class="mt-4"></div>
        <b-modal id="modal-selected" footerClass="p-2 border-top-0" :title="`${selectedEnvironment.displayedName}`" size="lg" scrollable hide-footer>

            <b-container fluid>

                <b-row>
                    <b-col>

                        <div>
                            <span class="data_label  mt-2">Categories: </span>
                            <span class="data_label">{{ selectedEnvironment.categories }}</span>
                        </div>
                        <div>
                            <span class="data_label  mt-2">Version: </span>
                            <span class="data_label">{{ selectedEnvironment.version }}</span>
                        </div>

                        <div>
                            <span class="data_label  mt-2">Extension: </span>
                            <span class="data_label">{{ selectedEnvironment.extensionId }}: {{ selectedEnvironment.extensionVersion }}</span>
                        </div>

                        <div>
                            <span class="data_label mt-2">Description: </span>
                            <span>{{selectedEnvironment.description}}</span>
                        </div>
                    </b-col>
                    <b-col>
                        <img :src="`${appConfig.API_URL}/api/env_preview_image/env_${selectedEnvironment.ident}`" alt="" style="max-height:200px;" />
                    </b-col>
                </b-row>
            </b-container>

            <div class="mt-4">
            </div>
            <h4>Environment Specs</h4>
            <b-container fluid>
                <div v-for="spec in selectedEnvironment.specs" :key="spec.ident">
                    <b-row>
                        <b-col cols=3 class="text-center  align-middle">
                            <div class="mt-2 mb-2">
                                <router-link :to="`/env_spec/view/${spec.ident}`">
                                    <b-card-img :src="`${appConfig.API_URL}/api/env_preview_image/${spec.ident}`" alt="No Image Found" style="width:100px;"></b-card-img>
                                </router-link>
                            </div>

                        </b-col>
                        <b-col>
                            <div>
                                <span>
                                    <router-link :to="`/env_spec/view/${spec.ident}`"> {{ spec.displayedName }}
                                    </router-link>
                                </span>
                            </div>
                            <div>
                                <span class="data_label mt-1">SpecId: </span>
                                <span>
                                    {{ spec.ident }}

                                </span>
                            </div>
                            <div>
                                <span class="data_label mt-1">Type: </span>
                                <span>{{spec.envType}}</span>
                            </div>

                            <div>
                                <span class="data_label mt-1">Tags: </span>
                                <span>{{spec.tags}}</span>
                            </div>
                            <div>
                                <span class="data_label mt-1">Description: </span>
                                <span>{{spec.description}}</span>
                            </div>
                        </b-col>

                        <b-col cols=2>

                            <div>
                                <b-button variant="primary" :to="`/task/new/agenttask/?envSpecId=${spec.ident}`" size="sm">Assign</b-button>

                            </div>

                        </b-col>
                    </b-row>
                    <div class="mt-1"></div>

                    <hr />
                </div>
            </b-container>

        </b-modal>

        <div class="mt-4"></div>

        <b-container fluid>
            <div v-if="$apollo.queries.environments.loading">Loading..</div>
            <div v-else>
                <b-row>
                    <b-col>
                        <b-form-input v-model="searchtext" placeholder="Search Environments" style="width:250px;" class='ml-auto'></b-form-input>
                    </b-col>
                </b-row>
                <div v-if="items.length > 0">

                    <div class="mt-4"></div>
                    <b-row>
                        <b-col class="d-flex flex-wrap justify-content-center  mb-4">
                            <span v-for="(item, idx) in items" v-bind:key="idx" class="m-2" @click="selectGroup(idx)" v-b-modal.modal-selected style="min-width:150px">
                                <EnvironmentCard :item="item"></EnvironmentCard>

                            </span>
                        </b-col>
                    </b-row>
                </div>
                <div v-else>{{ message }}</div>

            </div>
        </b-container>
        <br />
        <div class="mt-4"></div>
    </div>
    <HelpInfo contentId="envs"></HelpInfo>
</div>
</template>

<script>
// @ is an alias to /src
import gql from "graphql-tag";
import {
    appConfig
} from "../app.config";
import EnvironmentCard from "../components/EnvCardGroup.vue";
import {
    GET_ALL_ENVS
} from "../queries"

export default {
    name: "Env",
    components: {
        EnvironmentCard
    },
    apollo: {
        environments: GET_ALL_ENVS,
    },
    data() {
        return {
            environments: [],
            searchQuery: "",
            selectedEnvironment: {},
            message: "No Environments found.",
            appConfig,
            searchtext: ""
        };
    },
    methods: {
        selectGroup(idx) {
            this.selectedEnvironment = this.items[idx];

        },
    },

    computed: {
        items() {
            if (this.environments.length == 0) return [];
            else {
                let envs = this.environments.edges.map((v) => v.node).filter((v) => !v.disabled);
                if (this.searchtext != "") {
                    return envs.filter((v) =>
                        v.displayedName.toLowerCase().includes(this.searchtext.toLowerCase())
                    )
                } else {
                    return envs

                }
            }
        }
    },
    created() {
        //
    },
};
</script>

<style >
</style>
